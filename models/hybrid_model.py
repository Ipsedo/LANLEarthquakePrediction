import torch as th
import torch.nn as nn


n_ft_mul = 2.0


class ConvModelNoReduction(nn.Module):
    def __init__(self, in_channel):
        super(ConvModelNoReduction, self).__init__()

        self.seq_conv = nn.Sequential(
            nn.Conv1d(in_channel, in_channel * n_ft_mul, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.seq_conv(x)


class ConvModelWithReduction(nn.Module):

    def __init__(self, in_channel):
        super(ConvModelWithReduction, self).__init__()

        nb_channel_2 = int(in_channel * 4.0 / 3.0)

        self.__seq_conv = nn.Sequential(
            nn.Conv1d(in_channel, nb_channel_2, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(3, 3),
            nn.Conv1d(nb_channel_2, int(in_channel * n_ft_mul), kernel_size=9),
            nn.ReLU(),
            nn.MaxPool1d(5, 5)
        )

    def forward(self, x):
        return self.__seq_conv(x)


class RecModel(nn.Module):
    mul_hidden = 2

    def __init__(self, out_channel_conv, batch_size):
        super(RecModel, self).__init__()
        self.in_channel = out_channel_conv
        self.batch_size = batch_size

        self.lstm = nn.LSTM(self.in_channel, self.in_channel * RecModel.mul_hidden, batch_first=True)
        #self.fst_h = th.randn(1, self.batch_size, self.in_channel * RecModel.mul_hidden)
        #self.fst_x = th.randn(1, self.batch_size, self.in_channel * RecModel.mul_hidden)

    def __init_hiddens(self, batch_size):
        return (th.randn(1, batch_size, self.in_channel * RecModel.mul_hidden),
                th.randn(1, batch_size, self.in_channel * RecModel.mul_hidden))

    def forward(self, out_conv):
        out = out_conv.transpose(2, 1)

        #h, x = self.fst_h[:, :out.size(0), :], self.fst_x[:, :out.size(0), :]
        h, x = self.__init_hiddens(out.size(0))

        if next(self.parameters()).is_cuda:
            h, x = h.cuda(), x.cuda()

        out, _ = self.lstm(out, (h, x))

        return out


class HybridModel(nn.Module):
    def __init__(self, in_channel, batch_size, all_label):
        super(HybridModel, self).__init__()

        self.all_label = all_label

        if all_label:
            self.conv = ConvModelNoReduction(in_channel)
        else:
            self.conv = ConvModelWithReduction(in_channel)

        self.lstm = RecModel(int(in_channel * n_ft_mul), batch_size)

        size_hidden_lin = 17 * 1000

        self.lin_seq = nn.Sequential(
            nn.Linear(int(in_channel * n_ft_mul) * RecModel.mul_hidden, size_hidden_lin),
            nn.Dropout(0.3),
            nn.Linear(size_hidden_lin, 1)
        )

    def forward(self, x):
        out_conv = self.conv(x)
        out_lstm = self.lstm(out_conv)
        if not self.all_label:
            out_lstm = out_lstm[:, -1, :]
        out_lin = self.lin_seq(out_lstm)
        return out_lin.view(self.lstm.batch_size, -1)
