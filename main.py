#/usr/bin/env python

from data.load_csv import open_csv
from data.process import spectro_seg
import matplotlib.pyplot as plt


if __name__ == "__main__":
    res_root = "./res/"
    train_csv = res_root + "train.csv"
    x, y = open_csv(train_csv)
    f, t, Sxx = spectro_seg(x[0, :])

    print(Sxx.shape)


    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

