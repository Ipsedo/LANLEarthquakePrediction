import torch as th
import torch.nn as nn


class MyLSTM(nn.Module):
    def __init__(self, in_channel, all_label):
        super(MyLSTM, self).__init__()
        self.__in_channel = in_channel
        self.__all_label = all_label
        self.__lstm = nn.LSTM(in_channel, in_channel * 4, batch_first=True)
        self.__linear = nn.Linear(in_channel * 4, 1)

    def __init_hiddens(self, batch_size):
        return (th.randn(1, batch_size, self.__in_channel * 4),
                th.randn(1, batch_size, self.__in_channel * 4))

    def forward(self, data):
        batch_size = data.size(0)
        h, x = self.__init_hiddens(batch_size)

        if next(self.parameters()).is_cuda:
            h, x = h.cuda(), x.cuda()

        data = data.transpose(2, 1)

        out, _ = self.__lstm(data, (h, x))

        if not self.__all_label:
            out = out[:, -1, :]

        out = self.__linear(out)

        return out.view(batch_size, -1)
