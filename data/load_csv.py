import pandas as pd
import numpy as np


train_data_frame = 629145480
train_segmentsize = 2280


def open_csv(file_name):
    df = pd.read_csv(file_name, nrows=train_segmentsize)
    a = df.to_numpy().reshape(-1, train_segmentsize, 2)
    x, y = a[:, :, 0], a[:, :, 1]
    return x, y
