import torch as th
import torch.nn as nn


class ConvModel(nn.Module):
    def __init__(self, in_channel):
        super(ConvModel, self).__init__()
        self.in_channel = in_channel

        self.seq_conv = nn.Sequential(
            nn.Conv1d(self.in_channel, self.in_channel * 4, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(self.in_channel * 4, self.in_channel * 4 ** 2, kernel_size=7, stride=3),
            nn.ReLU(),
            nn.MaxPool1d(30, 30)
        )

    def forward(self, x):
        return self.seq_conv(x)


class RecModel(nn.Module):
    def __init__(self, out_channel_conv, batch_size):
        super(RecModel, self).__init__()
        self.in_channel = out_channel_conv
        self.batch_size = batch_size

        self.lstm = nn.LSTM(self.in_channel, self.in_channel * 2, batch_first=True)
        self.fst_h = th.randn(1, self.batch_size, self.in_channel * 2)
        self.fst_x = th.randn(1, self.batch_size, self.in_channel * 2)

    def forward(self, out_conv):
        out = out_conv.transpose(2, 1)
        h, x = self.fst_h[:, :out.size(0), :], self.fst_x[:, :out.size(0), :]
        if next(self.parameters()).is_cuda:
            h, x = h.cuda(), x.cuda()

        out, _ = self.lstm(out, (h, x))
        return out


class HybridModel(nn.Module):
    def __init__(self, in_channel, batch_size):
        super(HybridModel, self).__init__()

        self.conv = ConvModel(in_channel)
        self.lstm = RecModel(in_channel * 4 ** 2, batch_size)

        self.lin_seq = nn.Sequential(
            nn.Linear(in_channel * 2 * 4 ** 2, 1)
        )

    def forward(self, x):
        out_conv = self.conv(x)
        out_lstm = self.lstm(out_conv)
        out_lin = self.lin_seq(out_lstm[:, -1, :])
        return out_lin
