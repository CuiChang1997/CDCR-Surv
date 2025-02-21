from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch.utils.data import Dataset

class Dataset(Dataset):

    def __init__(self, X, y):

        self.X = X
        self.time = y[:, -1, 1].reshape(-1, 1)
        self.reason = y[:, -1, 0].reshape(-1, 1)
        self.label = y[:, -1, -1].reshape(-1, 1)


    def __getitem__(self, item):

        X_item = self.X[item]
        time_item = np.array(self.time[item])
        reason_item = np.array(self.reason[item])
        label_item = np.array(self.label[item])

        X_tensor = torch.from_numpy(X_item).float()
        time_tensor = torch.from_numpy(time_item).float()
        reason_tensor = torch.from_numpy(reason_item).float()
        label_item = torch.from_numpy(label_item).float()

        return X_tensor, time_tensor, reason_tensor, label_item

    def __len__(self):
        return self.X.shape[0]