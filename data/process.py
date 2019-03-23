import numpy as np
from scipy.signal import spectrogram


sampling_rate = 44e3


def spectro_seg(seg):
    return spectrogram(seg, nperseg=3, nfft=16)

