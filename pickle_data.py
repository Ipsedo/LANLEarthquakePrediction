from data.load_csv import open_multiple_csv
from data.process import split_signal_earthquake, omit_signal, fft_all, fft_window, n_fft
from os import mkdir
from os.path import exists

if __name__ == "__main__":
    res_root = "./res/"
    train_csv = res_root + "train.csv"

    x, y = open_multiple_csv(res_root + "train_splitted")

    print("X : {} {}, y : {} {}".format(x.shape, x.dtype, y.shape, y.dtype))

    x_list, y_list = split_signal_earthquake(x, y)
    print("Signal splitted (nb earthquake = %d" % (len(x_list)))

    nb_train = int(len(x_list) * 3 / 4)
    x_train, y_train = x_list[:nb_train], y_list[:nb_train]
    x_dev, y_dev = x_list[nb_train:], y_list[nb_train:]

    x_train, y_train = omit_signal(x_train, y_train)
    print("X_train : {}, y_train : {}".format(x_train.shape, y_train.shape))

    x_dev, y_dev = omit_signal(x_dev, y_dev)
    print("X_dev : {}, y_dev : {}".format(x_dev.shape, y_dev.shape))

    print("Data ommited / prepared for fft !")

    x_train, y_train = fft_all(x_train), y_train[:, -1]
    print("X_train : {}, y_train : {}".format(x_train.shape, y_train.shape))

    x_dev, y_dev = fft_all(x_dev), y_dev[:, -1]
    print("X_dev : {}, y_dev : {}".format(x_dev.shape, y_dev.shape))

    print("FFT finished !")

    # Pickle
    root_pickle = res_root + "pickle/"
    if not exists(root_pickle):
        mkdir(root_pickle)

    pickle_train_dir = root_pickle + "train/"
    if not exists(pickle_train_dir):
        mkdir(pickle_train_dir)

    file_prefix_name = "train_data_fftwindow-" + str(fft_window) + "_nfft-" + str(n_fft)
    pickle_data(x_train, y_train, pickle_train_dir, file_prefix_name)

    print("Train pickled !")

    pickle_dev_dir = root_pickle + "dev/"
    if not exists(pickle_dev_dir):
        mkdir(pickle_dev_dir)

    file_prefix_name = "dev_data_fftwindow-" + str(fft_window) + "_nfft-" + str(n_fft)
    pickle_data(x_dev, y_dev, pickle_dev_dir, file_prefix_name)

    print("Dev pickled !")
