from data.load_csv import open_multiple_csv
from data.process import split_signal_earthquake, omit_signal, fft_all, fft_window, n_fft, cwt_all, n_cwt, spectro_all, n_fft_calculated, n_overlap
from os import mkdir
from os.path import exists
from data.load_save_pickle import save_pickle
import numpy as np


def make_fft(x_t, y_t, x_d, y_d):
    all_label = False
    print("FFT with window size of %d and %d-frequencies will begin !" % (fft_window, n_fft))

    print("All label = {}".format(all_label))
    slice_y = np.arange(fft_window, 150000, step=fft_window) if all_label else -1

    x_t, y_t = fft_all(x_t), y_t[:, slice_y]
    print("X_train : {}, y_train : {}".format(x_t.shape, y_t.shape))

    x_d, y_d = fft_all(x_d), y_d[:, slice_y]
    print("X_dev : {}, y_dev : {}".format(x_d.shape, y_d.shape))

    print("FFT finished !")

    # Pickle
    data_name = "fftwindow-" + str(fft_window) + "_nfft-"\
                + str(n_fft) + "_calculated-" + str(n_fft_calculated) + "_alllabel-" + str(all_label)

    root_pickle = res_root + "pickle/"
    if not exists(root_pickle):
        mkdir(root_pickle)

    pickle_train_dir = root_pickle + "train_" + data_name + "/"
    if not exists(pickle_train_dir):
        mkdir(pickle_train_dir)

    file_prefix_name = "train_data_" + data_name
    save_pickle(x_t, y_t, pickle_train_dir, file_prefix_name)

    print("Train pickled !")

    pickle_dev_dir = root_pickle + "dev_" + data_name + "/"
    if not exists(pickle_dev_dir):
        mkdir(pickle_dev_dir)

    file_prefix_name = "dev_data_" + data_name
    save_pickle(x_d, y_d, pickle_dev_dir, file_prefix_name)

    print("Dev pickled !")


def make_cwt(x_t, y_t, x_d, y_d):
    x_t, y_t = cwt_all(x_t), y_t[:, -1]
    print("X_train : {}, y_train : {}".format(x_t.shape, y_t.shape))

    x_d, y_d = cwt_all(x_d), y_d[:, -1]
    print("X_dev : {}, y_dev : {}".format(x_d.shape, y_d.shape))

    print("CWT finished !")

    # Pickle
    data_name = "ncwt-" + str(n_cwt)

    root_pickle = res_root + "pickle/"
    if not exists(root_pickle):
        mkdir(root_pickle)

    pickle_train_dir = root_pickle + "train_" + data_name + "/"
    if not exists(pickle_train_dir):
        mkdir(pickle_train_dir)

    file_prefix_name = "train_data_" + data_name
    save_pickle(x_t, y_t, pickle_train_dir, file_prefix_name)

    print("Train pickled !")

    pickle_dev_dir = root_pickle + "dev_" + data_name + "/"
    if not exists(pickle_dev_dir):
        mkdir(pickle_dev_dir)

    file_prefix_name = "dev_data_" + data_name
    save_pickle(x_d, y_d, pickle_dev_dir, file_prefix_name)

    print("Dev pickled !")


def make_spectro(x_t, y_t, x_d, y_d):
    x_t, y_t = spectro_all(x_t), y_t[:, -1]
    print("X_train : {}, y_train : {}".format(x_t.shape, y_t.shape))

    x_d, y_d = spectro_all(x_d), y_d[:, -1]
    print("X_dev : {}, y_dev : {}".format(x_d.shape, y_d.shape))

    print("Spectrogram finished !")

    # Pickle
    data_name = "spectro_nfft-" + str(n_fft) + "_noverlap-" + str(n_overlap) + "_nperseg-" + str(n_fft)

    root_pickle = res_root + "pickle/"
    if not exists(root_pickle):
        mkdir(root_pickle)

    pickle_train_dir = root_pickle + "train_" + data_name + "/"
    if not exists(pickle_train_dir):
        mkdir(pickle_train_dir)

    file_prefix_name = "train_data_" + data_name
    save_pickle(x_t, y_t, pickle_train_dir, file_prefix_name)

    print("Train pickled !")

    pickle_dev_dir = root_pickle + "dev_" + data_name + "/"
    if not exists(pickle_dev_dir):
        mkdir(pickle_dev_dir)

    file_prefix_name = "dev_data_" + data_name
    save_pickle(x_d, y_d, pickle_dev_dir, file_prefix_name)

    print("Dev pickled !")


if __name__ == "__main__":
    res_root = "./res/"

    x, y = open_multiple_csv(res_root + "train_splitted")

    print("X : {} {}, y : {} {}".format(x.shape, x.dtype, y.shape, y.dtype))

    x_list, y_list = split_signal_earthquake(x, y)
    print("Signal splitted (nb earthquake = %d)" % (len(x_list)))

    nb_train = int(len(x_list) * 3 / 4)
    x_train, y_train = x_list[:nb_train], y_list[:nb_train]
    x_dev, y_dev = x_list[nb_train:], y_list[nb_train:]

    x_train, y_train = omit_signal(x_train, y_train)
    print("X_train : {}, y_train : {}".format(x_train.shape, y_train.shape))

    x_dev, y_dev = omit_signal(x_dev, y_dev)
    print("X_dev : {}, y_dev : {}".format(x_dev.shape, y_dev.shape))
    print("Data ommited !")

    #make_fft(x_train, y_train, x_dev, y_dev)
    #make_cwt(x_train, y_train, x_dev, y_dev)
    make_spectro(x_train, y_train, x_dev, y_dev)
