import copy
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm

from data_loaders.ehr_dataset import EHRDataset
from models.revised_modules import TransformerEncoderLayer


class LastEntryClf(Module):
    def __init__(
        self,
        d_model=512,
        nhead=1,
        nhid=100,
        nlayers=2,
        dropout=0.2,
        activation="relu",
        no_sa=False,
        no_ff=False,
        NN_params: dict = None,
        output_size=1,
    ):
        super(LastEntryClf, self).__init__()
        # variables from dataset
        NUM_LEN, CAT_LEN = EHRDataset.num_feature
        input_size = (NUM_LEN + CAT_LEN) * 2

        # embedding
        self.embedding_layer = nn.Linear(1, d_model)
        self.embedding_layer.apply(weights_init)

        # encoder
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=nhid, dropout=dropout,
            activation=activation, batch_first=True,
            no_sa=no_sa, no_ff=no_ff
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

        # NN decoder
        nn_dropout = NN_params["dropout_rate"]
        nn_activation = NN_params["activation"]
        nn_down_factor = NN_params["down_factor"]
        hidden_size = input_size // nn_down_factor
        if nn_activation == "relu":
            nn_act_fn = nn.ReLU()
        elif nn_activation == "gelu":
            nn_act_fn = nn.GELU()

        self.decoder = nn.Sequential(
            nn.Linear(input_size*d_model, hidden_size),
            nn_act_fn,
            nn.Dropout(nn_dropout),
            nn.Linear(hidden_size, output_size),
        )
        self.decoder.apply(weights_init)

    def forward(self, Xs):
        x_num, x_cat, x_num_mask, x_cat_mask = Xs
        x_feat = torch.cat([x_num, x_cat], 2)
        x_mask = torch.cat([x_num_mask, x_cat_mask], 2)
        x = torch.cat([x_feat, x_mask], 2)
        # x = x_feat
        # take out last entry
        x_last = x[:, -1, :]
        # shape: (N, input_size)
        x_last = torch.unsqueeze(x_last, 2)
        # shape: (N, input_size, 1)

        # embedding
        output = self.embedding_layer(x_last)
        # shape: (N, input_size, d_model)

        # encoder
        output = self.transformer_encoder(output)
        # shape: (N, input_size, d_model)
        input_size, d_model = output.size(1), output.size(2)

        # decoder
        output = output.view(-1, input_size*d_model)
        output = self.decoder(output)
        output = torch.sigmoid(output)

        return output


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)
