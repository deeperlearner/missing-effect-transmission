import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loaders.ehr_dataset import EHRDataset


class GRUNet(nn.Module):
    def __init__(
        self,
        CNN_params: dict,
        RNN_params: dict,
        NN_multi_params: dict,
        NN_params: dict,
        output_size=1,
    ):
        super(GRUNet, self).__init__()
        Time_LEN, Static_LEN = EHRDataset.num_feature
        self.input_size = Time_LEN * 2
        self.impute_method = EHRDataset.impute_method
        self.cat_type = EHRDataset.cat_type
        if self.cat_type == "norm_emb":
            self.cat_emb = nn.Embedding(2, 2, max_norm=1.0)

        # CNN: Conv1d
        self.cnns = None
        cnn_hiddens = CNN_params["hiddens"]
        params = {key: value for key, value in CNN_params.items() if key != "hiddens"}
        if len(cnn_hiddens) > 0:
            cnn_layers = layer_block(input_size, cnn_hiddens, CNN_layer, **params)
            self.cnns = nn.Sequential(*cnn_layers)
            rnn_input = cnn_hiddens[-1]
        else:
            rnn_input = input_size

        # RNN
        self.gru = None
        self.rnn_hidden_size = RNN_params["hidden_size"]
        self.rnn_num_layers = RNN_params["num_layers"]
        if self.rnn_hidden_size > 0:
            batch_norm = RNN_params["batch_norm"]
            dropout_rate = RNN_params["dropout_rate"]
            self.gru = nn.GRU(
                rnn_input,
                self.rnn_hidden_size,
                self.rnn_num_layers,
                batch_first=True,
                dropout=(0 if self.rnn_num_layers == 1 else dropout_rate),
            )
            nn_input = self.rnn_hidden_size
        else:
            # multi-dim NN
            nn_hiddens = NN_multi_params["hiddens"]
            params = {key: value for key, value in NN_multi_params.items() if key != "hiddens"}
            nn_layers = layer_block(input_size, nn_hiddens, NN_layer, **params)
            self.fcs_multi = nn.Sequential(*nn_layers)
            nn_input = nn_hiddens[-1]

        # NN decoder
        nn_dropout = NN_params["dropout_rate"]
        nn_activation = NN_params["activation"]
        nn_down_factor = NN_params["down_factor"]
        hidden_size = input_size // nn_down_factor
        if nn_activation == "relu":
            nn_act_fn = nn.ReLU()
        elif nn_activation == "gelu":
            nn_act_fn = nn.GELU()

        self.fcs = nn.Sequential(
            nn.Linear(self.rnn_hidden_size, hidden_size),
            nn_act_fn,
            nn.Dropout(nn_dropout),
            nn.Linear(hidden_size, output_size),
        )

        self.apply(weights_init)

    def forward(self, Xs):
        x_num, x_cat, x_num_mask, x_cat_mask = Xs
        if self.cat_type == "one_hot":
            x_cat = F.one_hot(x_cat)
        elif self.cat_type == "norm_emb":
            x_cat = self.cat_emb(x_cat)
        # squeeze dimension
        if self.cat_type == "one_hot" or self.cat_type == "norm_emb":
            emb_shape = x_cat.size()
            x_cat = x_cat.view(*emb_shape[:-2], -1)
            if self.impute_method == "zero":
                x_cat[~x_cat_mask.long()] = 0

        # x = torch.cat([x_num, x_cat, x_num_mask, x_cat_mask], 2)
        x = torch.cat([x_num, x_num_mask], 2)
        # print(x.size())
        # CNN
        if self.cnns is not None:
            # Turn into (batch_size, seq_len, input_size) into (batch_size, input_size, seq_len) for CNN
            x = x.transpose(1, 2)
            cnn_output = self.cnns(x)
            # Turn into (batch_size, input_size, seq_len) back into (batch_size, seq_len, input_size) for RNN
            rnn_input = cnn_output.transpose(1, 2)
        else:
            rnn_input = x
        # RNN
        if self.gru is not None:
            h = self.init_hidden(x.size(0))
            output, hidden = self.gru(rnn_input, h)
            output = F.relu(output)
        else:
            # multi-dim NN
            output = self.fcs_multi(x)
        last_vec = output[:, -1]

        # NN
        output = self.fcs(last_vec)
        output = torch.sigmoid(output)

        return output

    def init_hidden(self, batch_size):
        device = torch.device("cuda:0")
        return torch.zeros(
            self.rnn_num_layers, batch_size, self.rnn_hidden_size, device=device
        )


def CNN_layer(
    in_channels,
    out_channels,
    conv_kernel=3,
    conv_pad=0,
    batch_norm=False,
    activation="relu",
    dropout_rate=0,
    pool_kernel=3,
    pool_stride=3,
):
    layers = []
    layers.append(nn.Conv1d(in_channels, out_channels, conv_kernel, padding=conv_pad))
    if batch_norm:
        layers.append(nn.BatchNorm1d(out_channels))
    if activation == "relu":
        layers.append(nn.ReLU())
    if dropout_rate:
        layers.append(nn.Dropout(dropout_rate))
    layers.append(nn.AvgPool1d(pool_kernel, stride=pool_stride))
    return nn.Sequential(*layers)


def NN_layer(fc_in, fc_out, batch_norm=False, activation="relu", dropout_rate=0):
    layers = []
    layers.append(nn.Linear(fc_in, fc_out))
    if batch_norm:
        layers.append(nn.BatchNorm1d(out_channels))
    if activation == "relu":
        layers.append(nn.ReLU())
    elif activation == "gelu":
        layers.append(nn.GELU())
    if dropout_rate:
        layers.append(nn.Dropout(dropout_rate))
    return nn.Sequential(*layers)


def layer_block(in_layers, hidden_layers, layer_type, **kwargs):
    layers = []
    c_in = in_layers
    for h in hidden_layers:
        c_out = h
        layers.extend(layer_type(c_in, c_out, **kwargs))
        c_in = h
    return layers


def weights_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)
