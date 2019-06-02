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


RECORDING_TIME = 0.5
LOW_PASS_THRESHOLD = 0.075
DUDES = ['jackson', 'nicolas', 'theo', 'yweweler']
DUDES = ['jackson']
SPEAKER = 'fake_jackson'
WORD = '1'

FRAME_OVERLAP = 0.25

RECORDING_SAMPLING_RATE = 48000
SAMPLING_RATE = 8000
BUFFER_SIZE = 256
BUFFER_DISPLAY_SIZE = int(RECORDING_TIME * RECORDING_SAMPLING_RATE)
FFT_CAP = 20


if args.hifi:
    SAMPLING_RATE = 48000
    BUFFER_SIZE = 512
    BUFFER_DISPLAY_SIZE = BUFFER_SIZE
    FFT_CAP = 125

if args.lofi:
    SAMPLING_RATE = 44100
    BUFFER_SIZE = 256
    BUFFER_DISPLAY_SIZE = BUFFER_SIZE // 4
    FFT_CAP = 75
    args.downsample = 10

q = queue.Queue()
samples = np.array([])
paused = False

def process_sample_bufffer(samples):
    # Identify data
    identified = True
    test = generate_fingerprint(samples[0::(RECORDING_SAMPLING_RATE // SAMPLING_RATE)])
    result, error = identify_sample(master_fingerprints, test)

    # Plot fingerprints
    lines[1].set_ydata(test)
    lines[2].set_ydata(master_fingerprints[result])

    # Plot raw test audio wave
    lines[3].set_ydata(samples[:BUFFER_DISPLAY_SIZE])

    resultText.set_text(result)
    confidenceText.set_text('{0:.2f}'.format(error) if identified else '')

def audio_callback(indata, frames, time, status):
    # Fancy indexing with mapping creates a (necessary!) copy:
    q.put(indata[::args.downsample, mapping])

    global recordingWord
    dataPoints = indata[::args.downsample]
    for value in dataPoints:
        if not recordingWord and abs(value) >= LOW_PASS_THRESHOLD:
            recordingWord = True

    global samples
    recordingBankSize = int(RECORDING_TIME * RECORDING_SAMPLING_RATE)
    if recordingWord and not paused and samples.shape[0] < recordingBankSize:
        # Grow sample buffer to desired size
        n = min(dataPoints.shape[0], recordingBankSize - samples.shape[0])
        samples = np.append(samples, dataPoints[:n])
        if (samples.shape[0] == recordingBankSize):
            # Buffer is complete, process audio samples
            process_sample_bufffer(samples)

            if False:
                # Save recorded samples to wav file
                global recording_iteration
                save_wav_file(samples, WORD, SPEAKER, recording_iteration)
                recording_iteration += 1

            recordingWord = False
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
        if column < len(lines) - 3:
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
    for i in range(0, data.shape[0], size - int(size * FRAME_OVERLAP)):
        if i + size < data.shape[0]:
            segments.append(data[i:i + size])
    return segments

def get_fingerprint(path, plot = False):
    global lines

    # Get audio data and prepare for processing
    samplingFrequency, audioData = wavfile.read(path)
    audioData = np.array(audioData)

    if len(audioData.shape) > 1 and audioData.shape[1] > 1:
        audioData = audioData[:, 1]

    return generate_fingerprint(audioData, plot)

def generate_fingerprint(audioData, plot = False):
    audioData = np.interp(audioData, [np.amin(audioData), np.amax(audioData)] , [-1.0, 1.0])

    # Generate fingerprint by segmenting samples and averaging FFTs
    fingerprint = np.zeros(BUFFER_SIZE // 2)
    segments = segment_data(audioData, BUFFER_SIZE)
    for segment in segments:
        fft = scipy.fftpack.fft(segment)
        trimmedFFT = fft[:BUFFER_SIZE // 2];
        trimmedFFT = np.abs(trimmedFFT);
        fingerprint = fingerprint + (trimmedFFT / len(segments))

    if plot:
        fftData = np.interp(fingerprint, [0.0, FFT_CAP], [0, 1])
        lines[1].set_ydata(fftData)

    fingerprint[0] = 0
    return fingerprint

def get_master_fingerprint(number, plot = False):
    master_fingerprint = np.zeros(BUFFER_SIZE // 2)
    for dude in DUDES:
        for i in range(50):
            fingerprint = get_fingerprint('./recordings/%d_%s_%d.wav' % (number, dude, i))
            master_fingerprint = master_fingerprint + fingerprint
    master_fingerprint = master_fingerprint / (len(DUDES) * 50)

    if plot:
        fftData = np.interp(master_fingerprint, [0.0, FFT_CAP], [0, 1])
        lines[1].set_ydata(fftData)

    return master_fingerprint

def identify_sample(master_fingerprints, test, plot = False):
    min_error = float('Inf')
    result = -1
    for index, master in enumerate(master_fingerprints):
        error = 0
        for a, b in zip(master, test):
            error = error + abs(a - b)

        if error < min_error:
            min_error = error
            result = index

    if plot:
        plt.figure()
        plt.plot(test, label='TEST')
        plt.plot(master_fingerprints[result], label=result)
        plt.legend(loc=2)
        plt.show(block=False)

    return result, error

def get_accuracy(number, master_fingerprints):
    matches = np.zeros(10)
    for dude in DUDES:
        for i in range(1, 50):
            test = get_fingerprint('./recordings/%d_%s_%d.wav' % (number, dude, i))
            result, error = identify_sample(master_fingerprints, test)
            matches[result] = matches[result] + 1

    matches = matches / (len(DUDES) * 50)
    print(number, matches[number], matches)
    return matches[number]

def save_wav_file(samples, word, speaker, iteration):
    file_Name = word + '_' + speaker + '_' + str(iteration) + '.wav'
    wavfile.write(file_Name, RECORDING_SAMPLING_RATE, samples)

if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)

if args.samplerate is None:
    # device_info = sd.query_devices(args.device, 'input')
    # args.samplerate = device_info['default_samplerate']
    args.samplerate = RECORDING_SAMPLING_RATE

recordingWord = False
recording_iteration = 0

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

# FFT plot
fftLines = fftAxes.plot(xf, np.zeros((BUFFER_SIZE // 2)), xf, np.zeros((BUFFER_SIZE // 2)))
fftAxes.axis((0, args.samplerate / (2.0 * args.downsample), 0, FFT_CAP))
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
resultText = textAxes.text(0.5, 0.5, '5', size=36, ha='center', va='center', weight='bold')
confidenceText = textAxes.text(0.5, 0.25, '18.94', size=8, ha='center', va='center')
textAxes.set_xticks([])
textAxes.set_yticks([])
textAxes.axis('off')

figure.tight_layout(pad=0.5)
figure.canvas.mpl_connect('key_press_event', key_press)

lines = [samplingLines[0], fftLines[0], fftLines[1], bufferLines[0]]

# Generate master fingerprints for all numbers
master_fingerprints = []
for i in range(10):
    master_fingerprints.append(get_master_fingerprint(i))

# Test accuracy
for i in range(10):
    get_accuracy(i, master_fingerprints)

# Quick change short circuit
if not True:
    sys.exit(1)

animation = FuncAnimation(figure, update_plot, interval=args.interval)
stream = sd.InputStream(
    device=args.device, channels=max(args.channels),
    samplerate=args.samplerate, callback=audio_callback)
with stream:
    plt.show()
