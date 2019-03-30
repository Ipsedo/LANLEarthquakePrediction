from math import ceil
from tqdm import tqdm
import pickle as pkl
from os.path import join
from os import listdir
import numpy as np
from utils.thread import ThreadWithReturnValue


def pickle_data(x, y, root_dir, prefix_fine_name, nb_data_per_file=100):
    nb_file = ceil(x.shape[0] / nb_data_per_file)

    for i in tqdm(range(nb_file)):
        i_min, i_max = i * nb_data_per_file, \
                       (i + 1) * nb_data_per_file if (i + 1) * nb_data_per_file < x.shape[0] else x.shape[0]

        file_name = root_dir + prefix_fine_name + "_" + str(i) + ".p"

        x_batch, y_batch = x[i_min:i_max], y[i_min:i_max]
        pkl.dump((x_batch, y_batch), open(file_name, "wb"))


def load_pickle(root_folder, nb_thread=16):
    file_names = np.asarray([join(root_folder, f) for f in listdir(root_folder)])
    nb_batch = ceil(file_names.shape[0] / nb_thread)

    def run(file_name):
        return pkl.load(open(file_name, "rb"))

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
