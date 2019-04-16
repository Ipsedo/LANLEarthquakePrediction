import torch.nn as nn


class ConvModel_1(nn.Module):
    def __init__(self, nb_in_channel):
        super(ConvModel_1, self).__init__()

        self.nb_in_channel = nb_in_channel

        self.conv_seq = nn.Sequential(
            nn.Conv1d(nb_in_channel, nb_in_channel * 2, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(nb_in_channel * 2, nb_in_channel * 4, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(256, 256)
        )

        self.lin_seq = nn.Sequential(
            nn.Linear(nb_in_channel * 4 * 2, 1)
        )

    def forward(self, x):
        out = self.conv_seq(x).view(-1, self.nb_in_channel * 4 * 2)
        out = self.lin_seq(out)
        return out


class ConvModel_2(nn.Module):
    def __init__(self, in_channel):
        super(ConvModel_2, self).__init__()

        self.__in_channel = in_channel

        self.__seq_conv = nn.Sequential(
            nn.Conv1d(in_channel, in_channel * 6, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(in_channel * 6, in_channel * 16, kernel_size=7),
            nn.ReLU(),
            nn.Conv1d(in_channel * 16, in_channel * 24, kernel_size=9, stride=3),
            nn.BatchNorm1d(in_channel * 24),
            nn.MaxPool1d(100, 100)
        )

        self.__seq_lin = nn.Sequential(
            nn.Linear(in_channel * 24 * 3, 1),
        )

    def forward(self, x):
        out_conv = self.__seq_conv(x).view(-1, self.__in_channel * 24 * 3)
        out_lin = self.__seq_lin(out_conv)
        return out_lin
