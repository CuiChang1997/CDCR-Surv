from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pycox.evaluation import EvalSurv
import pandas as pd


def data_loader(dataset):
    path = './data/' + dataset

    data_train = pd.read_csv(path + '/train.csv', encoding='utf_8')
    data_sequence_train = data_train.values.reshape(-1, 5, data_train.shape[1])
    X_train = data_sequence_train[:, :, :-3]
    y_train = data_sequence_train[:, :, -3:]

    data_test = pd.read_csv(path + '/test.csv', encoding='utf_8')
    data_sequence_test = data_test.values.reshape(-1, 5, data_test.shape[1])
    X_test = data_sequence_test[:, :, :-3]
    y_test = data_sequence_test[:, :, -3:]

    return X_train, X_test, y_train, y_test


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Eval(object):
    def __init__(self, K_reasons, time_len, p, time, reason):
        self.K_reasons = K_reasons
        self.time_len = time_len
        self.F = p.reshape(p.shape[0], self.K_reasons, -1).cumsum(-1)
        self.S = 1 - self.F
        self.evals = []
        for i in range(self.K_reasons):
            S_k = self.S[:, i, :]
            S_k_df = pd.DataFrame(S_k.T.detach().cpu().numpy(), np.arange(time_len) + 1)

            eval = EvalSurv(S_k_df, time.cpu().numpy().reshape(-1), (reason==i+1).cpu().numpy().reshape(-1),
                            censor_surv='km')
            self.evals.append(eval)

    def c_index(self):
        c_index = []
        for i in range(self.K_reasons):
            c_index.append(self.evals[i].concordance_td())
        return np.mean(c_index)

    def brier_score(self):
        brier_score = []
        time_grid = np.arange(self.time_len) + 1
        for i in range(self.K_reasons):
            brier_score.append(self.evals[i].integrated_brier_score(time_grid))
        return brier_score
