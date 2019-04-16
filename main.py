#/usr/bin/env python

from data.load_save_pickle import load_pickle
from data.process import n_fft, fft_window
from models.conv_models import ConvModel_1, ConvModel_2
from models.hybrid_model import HybridModel
from models.recurrent_models import MyLSTM
from metrics.mean_absolute_error import mae
import torch.nn as nn
import torch as th
from math import ceil
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from time import sleep


if __name__ == "__main__":

    all_label = False

    root_pickle = "./res/pickle/"
    root_pickle_train = root_pickle + "train_spectro_nfft-1024_noverlap-896_nperseg-1024/"
    root_pickle_dev = root_pickle + "dev_spectro_nfft-1024_noverlap-896_nperseg-1024/"

    x_train, y_train = load_pickle(root_pickle_train)
    x_dev, y_dev = load_pickle(root_pickle_dev)

    print("Data loaded !")

    print(x_train.shape)
    print(y_train.shape)

    x_train = (x_train - x_train.mean()) / x_train.std()
    x_dev = (x_dev - x_dev.mean()) / x_dev.std()

    nb_epoch = 30

    batch_size = 8
    nb_batch = ceil(x_train.shape[0] / batch_size)

    n_features = x_train.shape[1]

    #m = ConvModel_2(n_features)
    m = HybridModel(n_features, batch_size, all_label)
    #m = MyLSTM(n_features, all_label)
    mse = nn.MSELoss()

    m.cuda()
    mse.cuda()

    optim = th.optim.Adagrad(m.parameters(), lr=3e-4)

    loss_values = []
    mae_values = []

    for e in range(nb_epoch):

        sum_loss = 0

        m.train()

        indices = np.arange(0, nb_batch)
        np.random.shuffle(indices)

        for i in tqdm(indices):
            i_min, i_max = i * batch_size, (i + 1) * batch_size \
                if (i + 1) * batch_size < x_train.shape[0] \
                else x_train.shape[0]

            x_batch, y_batch = th.Tensor(x_train[i_min:i_max]).cuda(), \
                               th.Tensor(y_train[i_min:i_max]).cuda()

            optim.zero_grad()

            out = m(x_batch)
            loss_v = mse(out.view(-1), y_batch.view(-1))

            loss_v.backward()
            optim.step()

            sum_loss += loss_v.item()

        sum_loss /= nb_batch
        loss_values.append(sum_loss)

        sleep(0.1)
        tqdm.write("Epoch %d, loss = %f" % (e, sum_loss))
        sleep(0.1)

        m.eval()

        batch_size_dev = batch_size
        nb_batch_dev = ceil(x_dev.shape[0] / batch_size_dev)

        y_dev_pred = th.zeros(y_dev.shape[0], y_dev.shape[1]).cuda() if all_label \
            else th.zeros(y_dev.shape[0]).cuda()

        with th.no_grad():
            for i in tqdm(range(nb_batch_dev)):
                i_min, i_max = i * batch_size_dev, (i + 1) * batch_size_dev \
                                if (i + 1) * batch_size_dev < x_dev.shape[0] \
                                else x_dev.shape[0]

                x_batch = th.tensor(x_dev[i_min:i_max]).float().cuda()

                if all_label:
                    y_dev_pred[i_min:i_max, :] = m(x_batch)
                else:
                    y_dev_pred[i_min:i_max] = m(x_batch).view(-1)

        score = mae(y_dev_pred.view(-1), th.Tensor(y_dev).view(-1).cuda())

        plt.plot(np.linspace(0, 20, 2000), np.linspace(0, 20, 2000), "r-")
        plt.scatter(y_dev, y_dev_pred.cpu().detach().numpy(), c="b", s=1e-1, marker="o")
        plt.xlabel("True value")
        plt.ylabel("Predicted value")
        plt.title("Epoch %d" % (e,))
        plt.legend()
        plt.show()

        mae_values.append(score)

        sleep(0.1)
        tqdm.write("Mean absolute error = %f" % (score,))
        sleep(0.1)

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

