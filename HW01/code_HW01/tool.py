import torch
from torch.utils.data import random_split
import numpy as np


def train_valid_split(data_set, valid_ratio, seed):
    train_size = int(len(data_set) * (1 - valid_ratio))
    valid_size = len(data_set) - train_size
    train_data, valid_data = random_split(data_set, [train_size, valid_size], torch.Generator().manual_seed(seed))
    return train_data, valid_data


def select_feat(train_data, valid_data, test_data, select_all=True):
    y_train, y_valid = np.array(train_data)[:, -1], np.array(valid_data)[:, -1]
    raw_x_train, raw_x_valid, raw_x_test = np.array(train_data)[:, :-1], np.array(valid_data)[:, :-1], np.array(
        test_data)
    feat_idx = list(range(raw_x_train.shape[1]))
    return np.array(raw_x_train)[:, feat_idx], np.array(raw_x_valid)[:, feat_idx], np.array(raw_x_test)[:,
                                                                                   feat_idx], y_train, y_valid
