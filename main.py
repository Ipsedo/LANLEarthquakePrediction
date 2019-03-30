#/usr/bin/env python

from data.load_save_pickle import load_pickle
from data.process import n_fft
from models.conv_models import ConvModel
from models.hybrid_model import HybridModel
from metrics.mean_absolute_error import mae
import torch.nn as nn
import torch as th
from math import ceil
from tqdm import tqdm
import sys
import numpy as np


if __name__ == "__main__":

    root_pickle = "./res/pickle/"
    root_pickle_train = root_pickle + "train/"
    root_pickle_dev = root_pickle + "dev/"

    x_train, y_train = load_pickle(root_pickle_train)
    x_dev, y_dev = load_pickle(root_pickle_dev)

    print("Data loaded !")

    print(x_train.shape)

    nb_epoch = 10

    batch_size = 8
    nb_batch = ceil(x_train.shape[0] / batch_size)

    m = ConvModel(n_fft)
    #m = HybridModel(n_fft, batch_size)
    mse = nn.MSELoss()

    m.cuda()
    mse.cuda()

    optim = th.optim.SGD(m.parameters(), lr=1e-4)

    for e in range(nb_epoch):

        sum_loss = 0

        sys.stdout.flush()

        m.train()

        indices = np.arange(0, nb_batch)
        np.random.shuffle(indices)

        for i in tqdm(indices):
            i_min, i_max = i * batch_size, (i + 1) * batch_size if (i + 1) * batch_size < x_train.shape[0] else x_train.shape[0]

            x_batch, y_batch = th.Tensor(x_train[i_min:i_max]).cuda(), th.Tensor(y_train[i_min:i_max]).cuda()

            optim.zero_grad()

            out = m(x_batch)
            loss_v = mse(out, y_batch)

            loss_v.backward()
            optim.step()

            sum_loss += loss_v.item()

        sys.stdout.flush()
        sum_loss /= nb_batch
        print("Epoch %d, loss = %f" % (e, sum_loss))

        sys.stdout.flush()

        m.eval()

        batch_size_dev = 2
        nb_batch_dev = ceil(x_dev.shape[0] / batch_size_dev)

        y_dev_pred = th.zeros(x_dev.shape[0]).cuda()

        for i in tqdm(range(nb_batch_dev)):
            i_min, i_max = i * batch_size_dev, \
                           (i + 1) * batch_size_dev if (i + 1) * batch_size_dev < x_dev.shape[0] else x_dev.shape[0]

            x_batch, y_batch = th.Tensor(x_dev[i_min:i_max]).cuda(), th.Tensor(y_dev[i_min:i_max]).cuda()

            y_dev_pred[i_min:i_max] = m(x_batch).view(-1)

        score = mae(y_dev_pred, th.Tensor(y_dev).cuda())

        sys.stdout.flush()

        print("Mean absolute error = %f" % (score,))
