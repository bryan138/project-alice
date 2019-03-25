#!/usr/bin/env python3

import argparse
import queue
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

SAMPLING_RATE = 4800
BUFFER_SIZE = 512
BUFFER_DISPLAY_SIZE = BUFFER_SIZE // 8

ID_THRESHOLD = 200
FTT_CAP = 25.0

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

def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def FFT(x):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:  # this cutoff should be optimized
        return DFT_slow(x)
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N / 2] * X_odd,
                               X_even + factor[N / 2:] * X_odd])

def FFT_vectorized(x):
    """A vectorized, non-recursive version of the Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if np.log2(N) % 1 > 0:
        raise ValueError("size of x must be a power of 2")

    # N_min here is equivalent to the stopping condition above,
    # and should be a power of 2
    N_min = min(N, 32)

    # Perform an O[N^2] DFT on all length-N_min sub-problems at once
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))

    # build-up each level of the recursive calculation all at once
    while X.shape[0] < N:
        X_even = X[:, :X.shape[1] / 2]
        X_odd = X[:, X.shape[1] / 2:]
        factor = np.exp(-1j * np.pi * np.arange(X.shape[0])
                        / X.shape[0])[:, None]
        X = np.vstack([X_even + factor * X_odd,
                       X_even - factor * X_odd])

    return X.ravel()

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):

    """Detect peaks in data based on their amplitude and other features.
    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).
    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.
    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's
    See this IPython Notebook [1]_.
    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)
    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)
    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)
    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indexes of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indexes by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind

def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()

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
mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1

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
    peaks = detect_peaks(trimmedFFT, mpd=15)
    peaks = sorted(peaks, key=lambda x: trimmedFFT[x])
    peakA = xf[peaks[-2]]
    peakB = xf[peaks[-1]]

    # Plot FFT
    fftData = np.interp(trimmedFFT, [0.0, FTT_CAP], [0, 1])
    lines[1].set_ydata(fftData)
    lines[1].set_markevery([peaks[-2], peaks[-1]])

    # Identify DTMF
    identified = False
    dtmfNumber = '--'
    confidence = float('Inf')
    for i, tone in enumerate(DTMF):
        difference = abs(peakA - tone[0]) + abs(peakB - tone[1])

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
        n = min(indata.shape[0], BUFFER_SIZE - samples.shape[0])
        samples = np.append(samples, indata[:n])

        if (samples.shape[0] == BUFFER_SIZE):
            # Buffer is complete, go to processing and clean up for next buffer
            process_sample_bufffer(samples)
            samples = np.array([])

def update_plot(frame):
    """
    This is called by matplotlib for each plot update.
    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.
    """

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

try:
    from matplotlib.animation import FuncAnimation
    import sounddevice as sd

    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)

    if args.samplerate is None:
        # device_info = sd.query_devices(args.device, 'input')
        # args.samplerate = device_info['default_samplerate']
        args.samplerate = SAMPLING_RATE

    print('SAMPLE RATE:', args.samplerate, 'DOWNSAMPLE:', args.downsample)

    length = int(args.window * args.samplerate / (1000 * args.downsample))
    plotdata = np.zeros((length, len(args.channels)))
    xf = np.linspace(0.0, SAMPLING_RATE / 2.0, BUFFER_SIZE // 2)

    figure = plt.figure()
    grid = plt.GridSpec(4, 4)
    samplingAxes = figure.add_subplot(grid[0, :])
    fftAxes = figure.add_subplot(grid[2:, :])
    bufferAxes = figure.add_subplot(grid[1, :3])
    textAxes = figure.add_subplot(grid[1, 3])

    # FTT plot
    fttLines = fftAxes.plot(xf, np.zeros((BUFFER_SIZE // 2)), marker='.', markerfacecolor='r', markeredgecolor='r')
    fftAxes.axis((0, SAMPLING_RATE / 2.0, 0, 1))
    fftAxes.set_xticks(np.linspace(0, SAMPLING_RATE // 2, 6))
    fftAxes.tick_params(left=False, labelleft=False, labelsize='x-small')

    # Sampling plot
    samplingLines = samplingAxes.plot(plotdata)
    samplingAxes.axis((0, len(plotdata), -1, 1))
    samplingAxes.set_yticks([0])
    samplingAxes.yaxis.grid(True)
    samplingAxes.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

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
    animation = FuncAnimation(figure, update_plot, interval=args.interval)
    stream = sd.InputStream(
        device=args.device, channels=max(args.channels),
        samplerate=args.samplerate, callback=audio_callback)
    with stream:
        plt.show()
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
