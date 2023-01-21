import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loaders.ehr_dataset import EHRDataset


class BertModel(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=1,
        nhid=100,
        nlayers=2,
        dropout=0.2,
        activation="relu",
        ignore_padding=False,
        weight_constraint=False,
        NN_params: dict = None,
        output_size=1,
    ):
        super(BertModel, self).__init__()
        # variables
        Time_LEN, Static_LEN = EHRDataset.num_feature
        self.input_size = Time_LEN * 2
        self.impute_method = EHRDataset.impute_method
        self.cat_type = EHRDataset.cat_type
        if self.cat_type == "norm_emb":
            self.cat_emb = nn.Embedding(2, 2, max_norm=1.0)
        self.d_model = d_model
        self.ignore_padding = ignore_padding
        self.weight_constraint = weight_constraint

        # embedding
        if weight_constraint:
            half_size = self.input_size // 2
            weight = torch.zeros(half_size, d_model)
            nn.init.normal_(weight, mean=0.0, std=0.02)
            self.weight = nn.Parameter(weight)
            self.gamma = nn.Parameter(torch.ones(1))
            self.bias = nn.Parameter(torch.zeros(d_model))
        else:
            self.embedding_layer = nn.Linear(self.input_size, d_model)
            self.embedding_layer.apply(weights_init)

        # bert encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=nhid, dropout=dropout,
            activation=activation, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

        # NN decoder
        nn_dropout = NN_params["dropout_rate"]
        nn_activation = NN_params["activation"]
        nn_down_factor = NN_params["down_factor"]
        hidden_size = d_model // nn_down_factor
        if nn_activation == "relu":
            nn_act_fn = nn.ReLU()
        elif nn_activation == "gelu":
            nn_act_fn = nn.GELU()

        self.decoder = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn_act_fn,
            nn.Dropout(nn_dropout),
            nn.Linear(hidden_size, output_size),
        )
        self.decoder.apply(weights_init)

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

        # x_feat = torch.cat([x_num, x_cat], 2)
        # x_mask = torch.cat([x_num_mask, x_cat_mask], 2)
        x_feat = x_num
        x_mask = x_num_mask
        x = torch.cat([x_feat, x_mask], 2)

        if self.ignore_padding:
            src_key_padding_mask = torch.sum(x_mask, 2) < 1.
        else:
            src_key_padding_mask = None

        # embedding
        if self.weight_constraint:
            embedding_layer = torch.cat([self.weight, self.gamma * self.weight])
            output = torch.matmul(x, embedding_layer)
            output = output + self.bias
        else:
            output = self.embedding_layer(x)
        # shape: (N, seq_len, d_model)

        # encoder
        output = self.transformer_encoder(output, src_key_padding_mask=src_key_padding_mask)
        last_vec = output[:, -1]

        # decoder
        output = self.decoder(last_vec)
        output = torch.sigmoid(output)

        return output


def weights_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0.0, std=0.02)
        if m.bias is not None:
            m.bias.data.zero_()
        # nn.init.xavier_normal_(m.weight.data)
        # nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.LayerNorm):
        m.bias.data.zero_()
        m.weight.data.fill_(1.0)
