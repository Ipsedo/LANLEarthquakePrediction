#/usr/bin/env python

from data.load_save_pickle import load_pickle
from data.process import n_fft, fft_window
from models.conv_models import ConvModel
from models.hybrid_model import HybridModel
from metrics.mean_absolute_error import mae
import torch.nn as nn
import torch as th
from math import ceil
from tqdm import tqdm
import sys
import numpy as np
import matplotlib.pyplot as plt
from time import sleep


if __name__ == "__main__":

    root_pickle = "./res/pickle/"
    root_pickle_train = root_pickle + "train_fftwindow-50_nfft-64/"
    root_pickle_dev = root_pickle + "dev_fftwindow-50_nfft-64/"

    x_train, y_train = load_pickle(root_pickle_train)
    x_dev, y_dev = load_pickle(root_pickle_dev)

    print("Data loaded !")

    print(x_train.shape)

    nb_epoch = 30

    batch_size = 8
    nb_batch = ceil(x_train.shape[0] / batch_size)

    #m = ConvModel(n_fft)
    m = HybridModel(n_fft, batch_size)
    mse = nn.MSELoss()

    m.cuda()
    mse.cuda()

    optim = th.optim.Adagrad(m.parameters(), lr=1e-4)

    loss_values = []
    mae_values = []

    for e in range(nb_epoch):

        sum_loss = 0

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

        sum_loss /= nb_batch
        loss_values.append(sum_loss)
        sleep(1e-1)
        tqdm.write("Epoch %d, loss = %f" % (e, sum_loss))

        m.eval()
        m.cpu()

        batch_size_dev = batch_size
        nb_batch_dev = ceil(x_dev.shape[0] / batch_size_dev)

        y_dev_pred = th.zeros(x_dev.shape[0])

        for i in tqdm(range(nb_batch_dev)):
            i_min, i_max = i * batch_size_dev, \
                           (i + 1) * batch_size_dev if (i + 1) * batch_size_dev < x_dev.shape[0] else x_dev.shape[0]

            x_batch = th.tensor(x_dev[i_min:i_max], requires_grad=False).float()

            y_dev_pred[i_min:i_max] = m(x_batch).view(-1)

        score = mae(y_dev_pred, th.Tensor(y_dev))

        plt.plot(np.linspace(0, 20, 2000), np.linspace(0, 20, 2000), "r-")
        plt.scatter(y_dev, y_dev_pred.detach().numpy(), c="b", s=10, marker="o")
        plt.xlabel("True value")
        plt.ylabel("Predicted value")
        plt.title("Epoch %d" % (e,))
        plt.legend()
        plt.show()

        mae_values.append(score)

        sleep(1e-1)
        tqdm.write("Mean absolute error = %f" % (score,))
        m.cuda()

    model_name = "hybrid model " + str(n_fft) + "-freq " + str(fft_window) + "-windows"

    plt.plot(range(nb_epoch), loss_values, c="b")
    plt.title(model_name + " - Loss values")
    plt.xlabel("Epoch")
    plt.ylabel("loss value (MSE)")
    plt.legend()
    plt.show()

    plt.plot(range(nb_epoch), mae_values, c="r")
    plt.title(model_name + " - MAE values")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error")
    plt.legend()
    plt.show()

