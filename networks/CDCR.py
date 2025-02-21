from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

np.random.seed(1234)
torch.cuda.manual_seed(1234)

class SharedSubNetwork(nn.Module):

    def __init__(self, dims, dropout=0, activation='ReLU', norm=True, output=True):
        super(SharedSubNetwork, self).__init__()
        self.dims = dims
        self.dropout = dropout
        self.norm = norm
        self.activation = activation
        self.output = output

        self.model = self._build_network()

    def _build_network(self):
        layers = []
        for i in range(len(self.dims)-2):
            layers.append(nn.Linear(self.dims[i], self.dims[i+1]))
            layers.append(eval('nn.{}()'.format(self.activation)))
            if self.norm: # adds batchnormalize layer
                layers.append(nn.BatchNorm1d(self.dims[i+1]))
            layers.append(nn.Dropout(self.dropout))
        # builds sequential network
        layers.append(nn.Linear(self.dims[-2], self.dims[-1]))
        if self.output == False:
            layers.append(eval('nn.{}()'.format(self.activation)))
            if self.norm:  # adds batchnormalize layer
                layers.append(nn.BatchNorm1d(self.dims[-1]))
            layers.append(nn.Dropout(self.dropout))
        return nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config["hidden_size"] % config[
            "num_of_attention_heads"] == 0, "The hidden size is not a multiple of the number of attention heads"

        self.num_attention_heads = config['num_of_attention_heads']
        self.attention_head_size = int(config['hidden_size'] / config['num_of_attention_heads'])
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config['hidden_size'], self.all_head_size)
        self.key = nn.Linear(config['hidden_size'], self.all_head_size)
        self.value = nn.Linear(config['hidden_size'], self.all_head_size)

        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(
            mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(
            mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,
                                                                         -2))
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs,
                                     value_layer)

        context_layer = context_layer.permute(0, 2, 1,
                                              3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
        self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        output = self.dense(context_layer)

        return output, (attention_scores.abs().squeeze(1) / attention_scores.abs().squeeze(1).sum(-1, True))


class LSTM_AE(nn.Module):

    def __init__(self, params):
        super(LSTM_AE, self).__init__()

        self.device = params['device']
        self.visit_len = params['visit_len']
        self.K_reasons = params['K_reasons']
        self.time_len = params['time_len']

        self.feature_size = params['feature_size']
        self.LSTM_hidden_size = params['LSTM_hidden_size']
        self.LSTM_num_layers = params['LSTM_num_layers']
        self.MLP_pred_dims = params['MLP_pred_dims']

        self.Encoder = nn.LSTM(self.feature_size,
                            self.LSTM_hidden_size,
                            self.LSTM_num_layers, batch_first=True)

        self.Decoder = nn.LSTM(self.LSTM_hidden_size,
                            self.LSTM_hidden_size,
                            self.LSTM_num_layers, batch_first=True)

        self.MLP_pred = SharedSubNetwork(self.MLP_pred_dims,
                                        norm=False,
                                        activation='Tanh',
                                        dropout=0)

    def forward(self, X):

        enc_out, (enc_h_state, enc_c_state) = self.Encoder(X)
        hidden = enc_h_state[-1, :, :]
        dec_input = hidden.unsqueeze(1)
        dec_out = torch.zeros_like(enc_out)
        hidden_dec = (torch.zeros_like(enc_h_state), torch.zeros_like(enc_c_state))
        for i in range(X.shape[1]):
            dec_input, hidden_dec = self.Decoder(dec_input, hidden_dec)
            dec_out[:, i, :] = dec_input[:, 0, :]
        pred_out = self.MLP_pred(dec_out)

        return hidden, pred_out

class CDCR_net(nn.Module):
    def __init__(self, params):
        super(CDCR_net, self).__init__()
        self.params = params
        self.device = params['device']
        self.visit_len = params['visit_len']
        self.K_reasons = params['K_reasons']
        self.time_len = params['time_len']
        self.MLP_cs_dims = params['MLP_cs_dims']
        self.cs_dropout = params['cs_dropout']
        self.att = params['att']
        if self.att == False:
            self.MLP_cs_dims[0] = params['K_reasons'] * params['LSTM_hidden_size'] + params['feature_size']
        elif self.att == True:
            self.MLP_cs_dims[0] = params['LSTM_hidden_size'] + params['feature_size']

        self.AE_list = nn.ModuleList(LSTM_AE(params) for i in range(self.K_reasons))
        self.MLP_cs_list = nn.ModuleList(SharedSubNetwork(self.MLP_cs_dims,
                                                          norm=True,
                                                          dropout=self.cs_dropout) for i in range(self.K_reasons))
        config = {
            "num_of_attention_heads": 1,
            "hidden_size": self.params['LSTM_hidden_size']
        }
        self.att_net = SelfAttention(config)
        self.weight = 0

    def forward(self, X):
        h_list = []
        X_pred_list = []
        for i in range(self.K_reasons):
            hidden, pred_out = self.AE_list[i](X)
            h_list.append(torch.div(hidden, torch.norm(hidden, 2, 1, True)))
            X_pred_list.append(pred_out.flip(1))

        p_output = torch.zeros([X.shape[0], self.K_reasons, self.time_len]).to(self.device)
        if self.att == False:
            CS_input = torch.cat((torch.stack(h_list, dim=1).reshape(X.shape[0], -1), X[:, -1, :]), dim=1)
            for i, MLP_cs in enumerate(self.MLP_cs_list):
                p_output[:, i, :] = MLP_cs(CS_input)

        elif self.att == True:
            h = torch.stack(h_list, 1)
            input, self.weight = self.att_net(h)
            for i, MLP_cs in enumerate(self.MLP_cs_list):
                CS_input = torch.cat((input[:, i, :], X[:, -1, :]), dim=1)
                p_output[:, i, :] = MLP_cs(CS_input)

        p = F.softmax(p_output.reshape(p_output.shape[0], -1), dim=1)
        return h_list, X_pred_list, p


class CDCR_loss(nn.Module):

    def __init__(self, params):
        super(CDCR_loss, self).__init__()
        self.device = params['device']
        self.K_reasons = params['K_reasons']
        self.time_len = params['time_len']

        self.sigma = params['sigma']
        self.tao = params['tao']
        self.alpha_rank = params['alpha_rank']
        self.alpha_pred = params['alpha_pred']
        self.alpha_cl = params['alpha_cl']
        self.alpha_reg = params['alpha_reg']


    def nll_loss(self, p, time, reason, label):
        F = p.reshape(p.shape[0], self.K_reasons, -1).cumsum(-1)

        # uncensored
        index_uncensored = ((reason - 1) * 10 + (time - 1)).relu().long()
        p_k_t = torch.gather(p, dim=1, index=index_uncensored)
        loss_uncensored = - (p_k_t + 1e-8).log().mul(label)

        # censored
        index_censored = (time - 1).relu().long().expand(p.shape[0], self.K_reasons).unsqueeze(-1)
        F_t = torch.gather(F, dim=2, index=index_censored)
        loss_censored = - ((1 - F_t.sum(1)) + 1e-8).log().mul(1 - label)

        return loss_uncensored + loss_censored

    def rank_loss(self, p, time, reason):
        F = p.reshape(p.shape[0], self.K_reasons, -1).cumsum(-1)
        index_i = (time - 1).relu().long().expand(p.shape[0], self.K_reasons).unsqueeze(-1)
        F_i = torch.gather(F, dim=2, index=index_i)


        t_onehot = torch.zeros([p.shape[0], self.time_len]).to(self.device).scatter(1, (time-1).long(), 1)

        t_nn = time.matmul(torch.ones_like(time).transpose(0, 1))
        t_diff = t_nn - t_nn.transpose(0, 1)
        t_diff[t_diff > 0] = 0
        t_diff[t_diff < 0] = 1

        loss = torch.zeros_like(time)
        for i in range(self.K_reasons):
            F_k = F[:, i, :]
            F_k_ij = F_k.matmul(t_onehot.transpose(0, 1)).transpose(0, 1)
            div = (- (F_i[:, i, :] - F_k_ij) / self.sigma).exp()
            loss_k = (reason == i+1).mul(t_diff).mul(div).mean(1, keepdim=True)
            loss += loss_k

        return loss

    def pred_loss(self, X, X_pred_list):

        loss = []

        for i in range(len(X_pred_list)):
            loss.append((X - X_pred_list[i]).norm(2, -1).sum(1, keepdim=True))

        loss = torch.stack(loss).sum(0)

        return loss

    def cause_cl_loss(self, h_list, reason, label):

        loss = []
        for i in range(len(h_list)):
            h = h_list[i]
            h_product = h.matmul(h.T)

            label_i = (reason == i+1).float()
            pos_index = (label_i.matmul(label_i.T) - torch.eye(label_i.shape[0]).to(self.device)).relu()
            neg_index = (1 - label_i.matmul(label_i.T)).mul(label.T).mul(label_i)
            loss_i = (((h_product.mul(pos_index)/self.tao).exp())/(h_product.mul(neg_index)/self.tao).exp().sum(1, keepdim=True)).sum(1, keepdim=True)
            loss.append(loss_i)

        loss = torch.stack(loss).sum(0)

        return loss

    def reg_loss(self, h_list):
        h_k = torch.stack(h_list, 1)
        h_k = torch.div(h_k, torch.norm(h_k, 2, 2, True))
        h_k_product = h_k.matmul(h_k.transpose(1, 2))
        loss = h_k_product.sum(-1).sum(-1, keepdim=True) / (len(h_list)**2)
        return loss


    def forward(self, X, h_list, X_pred_list, p, time, reason, label):

        loss_nll = self.nll_loss(p, time, reason, label)
        loss_rank = self.alpha_rank * self.rank_loss(p, time, reason)
        loss_pred = self.alpha_pred * self.pred_loss(X, X_pred_list)
        loss_cl = self.alpha_cl * self.cause_cl_loss(h_list, reason, label)
        loss_reg = self.alpha_reg * self.reg_loss(h_list)

        loss_total = loss_nll + loss_rank + loss_pred + loss_cl + loss_reg

        return loss_total
