import numpy as np
from scipy.signal import spectrogram


sampling_rate = 44e3


def spectro_seg(seg, n):
    return spectrogram(seg, nperseg=n, nfft=16)

