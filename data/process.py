import numpy as np
from scipy.signal import spectrogram
from scipy.fftpack import fft
from math import ceil, floor
from threading import Thread
from data.load_csv import train_segmentsize
from utils.thread import ThreadWithReturnValue
from tqdm import tqdm
import pickle as pkl
from scipy import signal

sampling_rate = 44e3

n_fft_calculated = 2048
n_fft = 2048
n_overlap = n_fft - 256

fft_window = 128

n_cwt = 64


def spectro_all(seg, nb_thread=64):

    res = []

    def run(s, idx):
        f, t, sxx = spectro_seg(s[idx, :])
        res.append(sxx)

    nb_batch = ceil(seg.shape[0] / nb_thread)

    for b in range(nb_batch):
        i_min, _i_max = b * nb_thread, (b + 1) * nb_thread if (b + 1) * nb_thread < seg.shape[0] else seg.shape[0]

        ths = []
        for i in range(i_min, _i_max):
            t = Thread(target=run, args=(seg, i))
            t.start()
            ths.append(t)

        for t in ths:
            t.join()

    return np.asarray(res)


def spectro_seg(seg):
    return spectrogram(seg, nfft=n_fft, noverlap=n_overlap, nperseg=n_fft)


def fft_all(segs, nb_thread=64):

    res = np.zeros((segs.shape[0], n_fft, int(segs.shape[1] / fft_window)))

    def run(idx):
        res[idx, :, :] = fft_seg(segs[idx, :])

    nb_batch = ceil(segs.shape[0] / nb_thread)

    for b in tqdm(range(nb_batch)):
        i_min, _i_max = b * nb_thread, (b + 1) * nb_thread \
                if (b + 1) * nb_thread < segs.shape[0] else segs.shape[0]

        ths = []
        for i in range(i_min, _i_max):
            t = Thread(target=run, args=(i,))
            t.start()
            ths.append(t)

        for t in ths:
            t.join()

    return res


def fft_seg(seg):
    res = []
    for i in range(int(train_segmentsize / fft_window)):
        r = np.fft.fft(seg[i * fft_window:(i + 1) * fft_window], n_fft_calculated)[:n_fft]
        r = np.sqrt(r.real ** 2 + r.imag ** 2)
        res.append(r)
    return np.transpose(np.asarray(res), (1, 0))


def split_signal_earthquake(signals, times):
    split = np.where(times[1:] > times[:-1])[0] + 1
    return np.split(signals, split), np.split(times, split)


def omit_signal(signal_list, times_list, seg_length=150000, omit_first=True):
    x, y = [], []
    for s, t in zip(signal_list, times_list):

        nb_seg = floor(s.shape[0] / seg_length)
        omit = s.shape[0] - nb_seg * seg_length

        new_s = s[omit:] if omit_first else s[:-omit]
        new_t = t[omit:] if omit_first else t[:-omit]

        x.append(new_s)
        y.append(new_t)

    x = np.concatenate(x).reshape(-1, seg_length)
    y = np.concatenate(y).reshape(-1, seg_length)
    return x, y


def cwt_seg(seg):
    return signal.cwt(seg, signal.ricker, np.arange(1, n_cwt + 1))


def cwt_all(segs, nb_thread=64):
    res = np.zeros((segs.shape[0], n_cwt, segs.shape[1]))

    def run(idx):
        res[idx, :, :] = cwt_seg(segs[idx, :])

    nb_batch = ceil(segs.shape[0] / nb_thread)

    for b in tqdm(range(nb_batch)):
        i_min, _i_max = b * nb_thread, (b + 1) * nb_thread if (b + 1) * nb_thread < segs.shape[0] else segs.shape[0]

        ths = []
        for i in range(i_min, _i_max):
            t = Thread(target=run, args=(i,))
            t.start()
            ths.append(t)

        for t in ths:
            t.join()

    return res
