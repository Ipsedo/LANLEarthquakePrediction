import torch.nn as nn


class ConvModel(nn.Module):
    def __init__(self, nb_in_channel):
        super(ConvModel, self).__init__()

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
