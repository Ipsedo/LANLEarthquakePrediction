import numpy as np
from scipy.signal import spectrogram
from scipy.fftpack import fft
from math import ceil, floor
from threading import Thread
from data.load_csv import train_segmentsize
from utils.thread import ThreadWithReturnValue
from tqdm import tqdm
import pickle as pkl

sampling_rate = 44e3


fft_window = 100

n_fft = 64


def spectro_all(seg, n, nb_thread=64):

    res = []

    def run(s, idx):
        f, t, sxx = spectro_seg(s[idx, :], n)
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


def spectro_seg(seg, n):
    return spectrogram(seg, nperseg=n, nfft=64)


def fft_all(seg, nb_thread=64):

    res = np.zeros((seg.shape[0], n_fft, int(seg.shape[1] / fft_window)))

    def run(idx):
        res[idx, :, :] = fft_seg(seg[idx, :])

    nb_batch = ceil(seg.shape[0] / nb_thread)

    for b in tqdm(range(nb_batch)):
        i_min, _i_max = b * nb_thread, (b + 1) * nb_thread if (b + 1) * nb_thread < seg.shape[0] else seg.shape[0]

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
        r = fft(seg[i * fft_window:(i + 1) * fft_window], n_fft)#np.fft.fft(seg[i * fft_window:(i + 1) * fft_window], n_fft)
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
