from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.optim as optim
import numpy as np
from lib.utils import data_loader, AverageMeter, Eval
from lib.dataset import Dataset
from networks.CDCR import CDCR_net, CDCR_loss

np.random.seed(1234)
torch.cuda.manual_seed(1234)

# 超参数
params = dict()
if torch.cuda.is_available():
    params['device'] = torch.device('cuda:0')
else:
    params['device'] = torch.device('cpu')

params['dataset'] = "mimic"

# training
params['optimizer'] = 'Adam'
params['epoch_num'] = 100
params['learning_rate'] = 1e-3
params['milestones'] = [50]
params['gamma'] = 0.1
params['batch_size'] = 256
params['weight_decay'] = 1e-3

params['tao'] = 0.7
params['sigma'] = 1.
params['alpha_rank'] = 1
params['alpha_pred'] = 0.1
params['alpha_cl'] = 0.01
params['alpha_reg'] = 0.1

params['visit_len'] = 5
params['K_reasons'] = 5
params['time_len'] = 10
params['feature_size'] = 40

params['LSTM_hidden_size'] = 100
params['LSTM_num_layers'] = 2
params['MLP_pred_dims'] = [100, 100, 100, 40]
params['MLP_cs_dims'] = [1, 100, 100, 10]
params['pred_dropout'] = 0
params['att_dropout'] = 0
params['cs_dropout'] = 0.5
params['att'] = True

class CDCR_Surv(object):

    def __init__(self, params):

        self.params = params
        self.device = params['device']
        self.K_reasons = params['K_reasons']
        self.time_len = params['time_len']

        # networks
        self.CDCR_net = CDCR_net(params).to(self.device)
        self.criterion = CDCR_loss(params).to(params['device'])

        self.optimizer = eval('optim.{}'.format(self.params['optimizer']))(self.CDCR_net.parameters(),
                                                                           lr=self.params['learning_rate'],
                                                                           weight_decay=self.params['weight_decay'])

        self.optimizer_pretrain = eval('optim.{}'.format(self.params['optimizer']))(self.CDCR_net.parameters(),
                                                                           lr=self.params['learning_rate'],
                                                                           weight_decay=self.params['weight_decay'])
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.params['milestones'],
                                                                 self.params['gamma'], last_epoch=-1)

    def train(self):
        losses = AverageMeter()
        self.CDCR_net.train()
        for step, [X, time, reason, label] in enumerate(self.train_loader):
            X = X.to(self.device)
            time = time.to(self.device)
            reason = reason.to(self.device)
            label = label.to(self.device)

            h_list, X_pred_list, p = self.CDCR_net(X)
            loss = self.criterion(X, h_list, X_pred_list, p, time, reason, label)

            loss = loss.mean()
            losses.update(loss, X.size(0))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return losses.avg

    def test(self, X_test, y_test):

        X = torch.from_numpy(X_test).float()
        time = torch.from_numpy(y_test[:, -1, 1].reshape(-1, 1)).float()
        reason = torch.from_numpy(y_test[:, -1, 0].reshape(-1, 1)).float()
        label = torch.from_numpy(y_test[:, -1, -1].reshape(-1, 1)).float()

        self.CDCR_net.eval()
        with torch.no_grad():
            X = X.to(self.device)
            time = time.to(self.device)
            reason = reason.to(self.device)
            label = label.to(self.device)

            h_list, X_pred_list, p = self.CDCR_net(X)
            eval = Eval(self.K_reasons, self.time_len, p, time, reason)
            c_index = eval.c_index()
            return c_index

    def fit(self, X_train, y_train):
        self.train_dataset = Dataset(X_train, y_train)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=params['batch_size'], shuffle=True)

        # train
        for epoch in range(self.params['epoch_num']):
            loss = self.train()
            print('epoch:', epoch, 'loss:', loss.item())
            self.lr_scheduler.step()


if __name__ == '__main__':

    X_train, X_test, y_train, y_test = data_loader(params['dataset'])
    CDCR_Surv = CDCR_Surv(params)
    CDCR_Surv.fit(X_train, y_train)
    c_index = CDCR_Surv.test(X_test, y_test)
    print('mean C-index:', c_index)


