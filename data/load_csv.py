import pandas as pd
import numpy as np
from math import ceil
from os import listdir
from os.path import join
from tqdm import tqdm
from utils.thread import ThreadWithReturnValue


train_segmentsize = 150000
train_data_frame = train_segmentsize * 4000


def open_csv(file_name):
    df = pd.read_csv(file_name, nrows=train_data_frame, header='infer', engine='c')
    a = df.to_numpy().reshape(-1, train_segmentsize, 2)
    x, y = a[:, :, 0], a[:, :, 1]
    return x, y


def open_csv_2(file_name):
    df = pd.read_csv(file_name, header='infer', engine='c')
    a = df.to_numpy().reshape(-1, 2)
    x, y = a[:, 0], a[:, 1]
    return x, y


def open_multiple_csv(root_folder, nb_thread=16):

    def run(file_name):
        df = pd.read_csv(file_name, header=None)
        a = df.to_numpy().reshape(-1, 2)
        return a[:, 0], a[:, 1]

    file_names = np.sort(np.asarray([join(root_folder, f) for f in listdir(root_folder)]))
    nb_batch = ceil(file_names.shape[0] / nb_thread)

    x, y = [], []
    for b in tqdm(range(nb_batch)):
        i_min, i_max = b * nb_thread, \
                       (b + 1) * nb_thread if (b + 1) * nb_thread < file_names.shape[0] else file_names.shape[0]

        threads = []
        for i in range(i_min, i_max):
            t = ThreadWithReturnValue(target=run, args=(file_names[i],))
            t.start()
            threads.append(t)

        for t in threads:
            x_res, y_res = t.join()
            x.append(x_res), y.append(y_res)

    return np.concatenate(x, axis=0), np.concatenate(y, axis=0)
