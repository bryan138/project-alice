#!/usr/bin/env python3

import argparse
import queue
import sys
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import scipy.fftpack
from scipy.io import wavfile
from scipy.signal import find_peaks
from matplotlib.animation import FuncAnimation

def int_or_str(text):
    try:
        return int(text)
    except ValueError:
        return text

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-w', '--window', type=float, default=200, metavar='DURATION',
    help='visible time slot (default: %(default)s ms)')
parser.add_argument(
    '-i', '--interval', type=float, default=30,
    help='minimum time between plot updates (default: %(default)s ms)')
parser.add_argument(
    '-b', '--blocksize', type=int, help='block size (in samples)')
parser.add_argument(
    '-hf', '--hifi', action='store_true', help='enable high fidelity mode')
parser.add_argument(
    '-lf', '--lofi', action='store_true', help='enable low fidelity mode')
parser.add_argument(
    '-r', '--samplerate', type=float, help='sampling rate of audio device')
parser.add_argument(
    '-n', '--downsample', type=int, default=1, metavar='N',
    help='display every Nth sample (default: %(default)s)')
parser.add_argument(
    'channels', type=int, default=[1], nargs='*', metavar='CHANNEL',
    help='input channels to plot (default: the first)')
args = parser.parse_args()
if any(c < 1 for c in args.channels):
    parser.error('argument CHANNEL: must be >= 1')
mapping = [c - 1 for c in args.channels]

SAMPLING_RATE = 8000
BUFFER_SIZE = 256
BUFFER_DISPLAY_SIZE = BUFFER_SIZE
FTT_CAP = 25

if args.hifi:
    SAMPLING_RATE = 48000
    BUFFER_SIZE = 512
    BUFFER_DISPLAY_SIZE = BUFFER_SIZE
    FTT_CAP = 125

if args.lofi:
    SAMPLING_RATE = 44100
    BUFFER_SIZE = 256
    BUFFER_DISPLAY_SIZE = BUFFER_SIZE // 4
    FTT_CAP = 75
    args.downsample = 10

ID_THRESHOLD = 200

DTMF = [
    [941, 1336], #0
    [697, 1209], #1
    [697, 1336], #2
    [697, 1477], #3
    [770, 1209], #4
    [770, 1336], #5
    [770, 1477], #6
    [852, 1209], #7
    [852, 1336], #8
    [852, 1477], #9
    [941, 1209], #*
    [941, 1477], ##
    [697, 1633], #A
    [770, 1633], #B
    [852, 1633], #C
    [941, 1633], #D
]

DTMF_CODE = [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '*', '#', 'A', 'B', 'C', 'D' ]

q = queue.Queue()
samples = np.array([])
paused = False

def process_sample_bufffer(samples):
    global fft

    # Plot buffer
    lines[2].set_ydata(samples[:BUFFER_DISPLAY_SIZE])

    # Compute FFT
    fft = scipy.fftpack.fft(samples)
    trimmedFFT = np.abs(fft[:BUFFER_SIZE // 2]);

    # Get FFT peaks
    frequencyStep = (args.samplerate / (2 * args.downsample)) / (BUFFER_SIZE / 2)
    peakDistance = 256 / frequencyStep
    peaks = find_peaks(trimmedFFT, distance=peakDistance)[0]
    peaks = peaks[peaks > (256 / frequencyStep)]
    peaks = sorted(peaks, key=lambda x: trimmedFFT[x], reverse=True)
    peakA = min(peaks[0], peaks[1])
    peakB = max(peaks[0], peaks[1])
    peakAF = xf[peakA]
    peakBF = xf[peakB]

    # Plot FFT
    fftData = np.interp(trimmedFFT, [0.0, FTT_CAP], [0, 1])
    lines[1].set_ydata(fftData)
    lines[1].set_markevery([peakA, peakB])

    # Update peak frequency labels
    fftTextA.set_text('{}Hz'.format(int(peakAF)))
    if fftData[peakA] > 0.95:
        offset = (BUFFER_SIZE / (-80 if args.hifi else 250)) * frequencyStep
        fftTextA.set_x(peakAF + offset)
        fftTextA.set_y(0.95)
        fftTextA.set_horizontalalignment('left')
    else:
        fftTextA.set_x(peakAF)
        fftTextA.set_y(fftData[peakA] + 0.035)
        fftTextA.set_horizontalalignment('center')

    fftTextB.set_text('{}Hz'.format(int(peakBF)))
    if fftData[peakB] > 0.95:
        fftTextB.set_x(peakBF + ((BUFFER_SIZE / 250) * frequencyStep))
        fftTextB.set_y(0.95)
        fftTextB.set_horizontalalignment('left')
    else:
        fftTextB.set_x(peakBF)
        fftTextB.set_y(fftData[peakB] + 0.035)
        fftTextB.set_horizontalalignment('center')

    # Identify DTMF
    identified = False
    dtmfNumber = '--'
    confidence = float('Inf')
    for i, tone in enumerate(DTMF):
        difference = abs(peakAF - tone[0]) + abs(peakBF - tone[1])

        if (difference < ID_THRESHOLD and difference < confidence):
            identified = True
            dtmfNumber = DTMF_CODE[i]
            confidence = difference

    dtmfText.set_text(dtmfNumber)
    confidenceText.set_text('{0:.2f}'.format(confidence) if identified else '')

def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    # Fancy indexing with mapping creates a (necessary!) copy:
    q.put(indata[::args.downsample, mapping])

    global samples
    if not paused and samples.shape[0] < BUFFER_SIZE:
        # Grow sample buffer to desired size
        dataPoints = indata[::args.downsample]
        n = min(dataPoints.shape[0], BUFFER_SIZE - samples.shape[0])
        samples = np.append(samples, dataPoints[:n])

        if (samples.shape[0] == BUFFER_SIZE):
            # Buffer is complete, go to processing and clean up for next buffer
            process_sample_bufffer(samples)
            samples = np.array([])

def update_plot(frame):
    global plotdata
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data

    for column, line in enumerate(lines):
        if column < len(lines) - 2:
            line.set_ydata(plotdata[:, column])
    return lines

def key_press(event):
    global paused
    sys.stdout.flush()
    if event.key == ' ':
        paused = not paused
        samples = np.array([])

def segment_data(data, size):
    segments = []
    for i in range(0, data.shape[0], size):
        if i + size < data.shape[0]:
            segments.append(data[i:i + size])
    return segments

def get_fingerprint(path, plot = False):
    global lines

    # Get audio data and prepare for processing
    samplingFrequency, audioData = wavfile.read(path)
    audioData = np.array(audioData)
    audioData = np.interp(audioData, [np.amin(audioData), np.amax(audioData)] , [-1.0, 1.0])

    # Generate fingerprint by segmenting samples and averaging FFTs
    fingerprint = np.zeros(BUFFER_SIZE // 2)
    segments = segment_data(audioData, BUFFER_SIZE)
    for segment in segments:
        fft = scipy.fftpack.fft(segment)
        trimmedFFT = np.abs(fft[:BUFFER_SIZE // 2]);
        fingerprint = fingerprint + (trimmedFFT / len(segments))

    if plot:
        fftData = np.interp(fingerprint, [0.0, FTT_CAP], [0, 1])
        lines[1].set_ydata(fftData)

    return fingerprint

def get_master_fingerprint(number, plot = False):
    master_fingerprint = np.zeros(BUFFER_SIZE // 2)
    for i in range(50):
        fingerprint = get_fingerprint('./recordings/%d_jackson_%d.wav' % (number, i))
        master_fingerprint = master_fingerprint + (fingerprint / 50)

    if plot:
        fftData = np.interp(master_fingerprint, [0.0, FTT_CAP], [0, 1])
        lines[1].set_ydata(fftData)

    return master_fingerprint

if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)

if args.samplerate is None:
    # device_info = sd.query_devices(args.device, 'input')
    # args.samplerate = device_info['default_samplerate']
    args.samplerate = SAMPLING_RATE

plt.rcParams['toolbar'] = 'None'
length = int(args.window * args.samplerate / (1000 * args.downsample))
plotdata = np.zeros((length, len(args.channels)))
xf = np.linspace(0.0, args.samplerate / (2.0 * args.downsample), BUFFER_SIZE // 2)

figure = plt.figure('FQT 9000')
grid = plt.GridSpec(4, 4)
samplingAxes = figure.add_subplot(grid[0, :])
fftAxes = figure.add_subplot(grid[2:, :])
bufferAxes = figure.add_subplot(grid[1, :3])
textAxes = figure.add_subplot(grid[1, 3])

# FTT plot
fttLines = fftAxes.plot(xf, np.zeros((BUFFER_SIZE // 2)))
fftTextA = fftAxes.text(0, 0, '', size=7, ha='center', va='center')
fftTextB = fftAxes.text(0, 0, '', size=7, ha='center', va='center')
fftAxes.axis((0, args.samplerate / (2.0 * args.downsample), 0, 1))
fftAxes.set_xticks(np.linspace(0, (args.samplerate / args.downsample) // 2, 6))
fftAxes.tick_params(left=False, labelleft=False, labelsize='x-small')

# Sampling plot
samplingLines = samplingAxes.plot(plotdata)
samplingAxes.axis((0, len(plotdata), -1, 1))
samplingAxes.set_yticks([0])
samplingAxes.yaxis.grid(True)
samplingAxes.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
samplingAxes.set_title('Sample Rate: {}Hz'.format(args.samplerate // args.downsample), loc='left', fontsize=7)
samplingAxes.set_title('Fast Quijas Transformer 9000', fontweight='bold', fontsize=14)
samplingAxes.set_title('Buffer Size: {}'.format(BUFFER_SIZE), loc='right', fontsize=7)

# Buffer plot
bufferLines = bufferAxes.plot(np.zeros(BUFFER_DISPLAY_SIZE))
bufferAxes.axis((0, BUFFER_DISPLAY_SIZE - 1, -1, 1))
bufferAxes.set_yticks([0])
bufferAxes.yaxis.grid(True)
bufferAxes.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

# Text plot
dtmfText = textAxes.text(0.5, 0.5, '5', size=36, ha='center', va='center', weight='bold')
confidenceText = textAxes.text(0.5, 0.25, '18.94', size=8, ha='center', va='center')
textAxes.set_xticks([])
textAxes.set_yticks([])
textAxes.axis('off')

figure.tight_layout(pad=0.5)
figure.canvas.mpl_connect('key_press_event', key_press)

lines = [samplingLines[0], fttLines[0], bufferLines[0]]

master_fingerprints = []
for i in range(10):
    master_fingerprints.append(get_master_fingerprint(i))

animation = FuncAnimation(figure, update_plot, interval=args.interval)
stream = sd.InputStream(
    device=args.device, channels=max(args.channels),
    samplerate=args.samplerate, callback=audio_callback)
with stream:
    plt.show()
