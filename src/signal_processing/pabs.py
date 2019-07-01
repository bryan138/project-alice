#!/usr/bin/env python3

import argparse
import queue
import numpy as np
import sounddevice as sd
import scipy.fftpack
from scipy.io import wavfile
import time

RECORDING_SAMPLING_RATE = 44100
SAMPLING_RATE = 8000
BUFFER_SIZE = 256
RECORDING_TIME = 1.0
LOW_PASS_THRESHOLD = 0.075
DUDES = ['pepe']
TRAINING_WORDS = ['si', 'el', 'no','perro']
BUFFER_DISPLAY_SIZE = int(RECORDING_TIME * RECORDING_SAMPLING_RATE)

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

q = queue.Queue()
samples = np.array([])

def audio_callback(indata, frames, time, status):
    q.put(indata[::args.downsample, mapping])

    global recordingWord
    dataPoints = indata[::args.downsample]
    for value in dataPoints:
        if not recordingWord and abs(value) >= LOW_PASS_THRESHOLD:
            recordingWord = True

    global samples
    recordingBankSize = int(RECORDING_TIME * RECORDING_SAMPLING_RATE)
    if recordingWord and samples.shape[0] < recordingBankSize:
        n = min(dataPoints.shape[0], recordingBankSize - samples.shape[0])
        samples = np.append(samples, dataPoints[:n])
        if (samples.shape[0] == recordingBankSize):
            test = generate_fingerprint(samples[0::(RECORDING_SAMPLING_RATE // SAMPLING_RATE)])
            result, index, error = identify_sample(master_fingerprints, test)
            print('Dijo:', result)

            recordingWord = False
            samples = np.array([])

def segment_data(data, size):
    segments = []
    for i in range(0, data.shape[0], size - int(size * 0.25)):
        segment = data[i:min(data.shape[0], i + size)]
        segment = np.append(segment, np.zeros(size -  segment.shape[0]))
        segments.append(segment)
    return segments

def get_fingerprint(path, plot = False):
    global lines

    try:
        samplingFrequency, audioData = wavfile.read(path)
        audioData = np.array(audioData)

        if len(audioData.shape) > 1 and audioData.shape[1] > 1:
            audioData = audioData[:, 1]

        audioData = audioData[0::6]
        return generate_fingerprint(audioData, plot)

    except:
        return None

def generate_fingerprint(audioData, plot = False):
    audioData = np.interp(audioData, [np.amin(audioData), np.amax(audioData)] , [-1.0, 1.0])

    fingerprint = np.zeros(BUFFER_SIZE // 2)
    segments = segment_data(audioData, BUFFER_SIZE)
    for segment in segments:
        fft = scipy.fftpack.fft(segment)
        trimmedFFT = fft[:BUFFER_SIZE // 2];
        trimmedFFT = np.abs(trimmedFFT);
        fingerprint = fingerprint + (trimmedFFT / len(segments))

    fingerprint[0] = 0
    return fingerprint

def get_master_fingerprint(word, plot = False):
    wordCount = 0
    master_fingerprint = np.zeros(BUFFER_SIZE // 2)
    for dude in DUDES:
        for i in range(50):
            fingerprint = get_fingerprint('./recordings/%s_%s_%d.wav' % (word, dude, i))
            if fingerprint is None:
                continue

            wordCount += 1
            master_fingerprint = master_fingerprint + fingerprint

    master_fingerprint = master_fingerprint / (len(DUDES) * wordCount)

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

    return TRAINING_WORDS[result], index, error

if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)

if args.samplerate is None:
    args.samplerate = RECORDING_SAMPLING_RATE

recordingWord = False

master_fingerprints = []
for word in TRAINING_WORDS:
    master_fingerprints.append(get_master_fingerprint(word))

stream = sd.InputStream(
    device=args.device, channels=max(args.channels),
    samplerate=args.samplerate, callback=audio_callback)

with stream:
    time.sleep(999999)
