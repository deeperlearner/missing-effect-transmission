import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loaders.ehr_dataset import EHRDataset


class TransformerModel(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=1,
        n_enc=6,
        n_dec=6,
        nhid=100,
        dropout=0.2,
        activation="relu",
        ignore_padding=False,
        weight_constraint=False,
        output_size=1,
    ):
        super(TransformerModel, self).__init__()
        # variables
        NUM_LEN, CAT_LEN = EHRDataset.num_feature
        self.input_size = (NUM_LEN + CAT_LEN) * 2
        self.impute_method = EHRDataset.impute_method
        self.cat_type = EHRDataset.cat_type
        if self.cat_type == "norm_emb":
            self.cat_emb = nn.Embedding(2, 2, max_norm=1.0)
        self.d_model = d_model
        self.ignore_padding = ignore_padding
        self.weight_constraint = weight_constraint

        # input embedding
        if weight_constraint:
            half_size = self.input_size // 2
            weight_in = torch.zeros(half_size, d_model)
            nn.init.normal_(weight_in, mean=0.0, std=0.02)
            self.weight_in = nn.Parameter(weight_in)
            self.gamma_in = nn.Parameter(torch.ones(1))
            self.bias_in = nn.Parameter(torch.zeros(d_model))
        else:
            self.embedding_layer_in = nn.Linear(self.input_size, d_model)
            self.embedding_layer_in.apply(weights_init)
        # output embedding (shifted right)
        if weight_constraint:
            half_size = self.input_size // 2
            weight_out = torch.zeros(half_size, d_model)
            nn.init.normal_(weight_out, mean=0.0, std=0.02)
            self.weight_out = nn.Parameter(weight_out)
            self.gamma_out = nn.Parameter(torch.ones(1))
            self.bias_out = nn.Parameter(torch.zeros(d_model))
        else:
            self.embedding_layer_out = nn.Linear(self.input_size, d_model)
            self.embedding_layer_out.apply(weights_init)

        # Transformer
        self.transformer = nn.Transformer(
            d_model, nhead, num_encoder_layers=n_enc, num_decoder_layers=n_dec,
            dim_feedforward=nhid, dropout=dropout,
            activation=activation, batch_first=True
        )

        # decoder
        feat_dim = self.input_size // 2
        self.decoder = nn.Linear(d_model, feat_dim)
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

        x_feat = torch.cat([x_num, x_cat], 2)
        x_mask = torch.cat([x_num_mask, x_cat_mask], 2)
        x = torch.cat([x_feat, x_mask], 2)

        x_shift = torch.roll(x, 1, dims=1)
        x_shift[:, 0, :] = 0
        x_mask_shift = torch.roll(x_mask, 1, dims=1)
        x_mask_shift[:, 0, :] = 0

        # sz = x_shift.size(1)
        # tgt_mask = self.transformer.generate_square_subsequent_mask(sz)
        # device = torch.device("cuda:0")
        # tgt_mask = tgt_mask.to(device)
        if self.ignore_padding:
            src_key_padding_mask = torch.sum(x_mask, 2) < 1.
            tgt_key_padding_mask = torch.sum(x_mask_shift, 2) < 1.
        else:
            src_key_padding_mask = None
            tgt_key_padding_mask = None

        # input embedding
        if self.weight_constraint:
            embedding_layer_in = torch.cat([self.weight_in, self.gamma_in * self.weight_in])
            x_in = torch.matmul(x, embedding_layer_in)
            x_in = x_in + self.bias_in
        else:
            x_in = self.embedding_layer_in(x)
        # output embedding (shifted right)
        if self.weight_constraint:
            embedding_layer_out = torch.cat([self.weight_out, self.gamma_out * self.weight_out])
            x_out = torch.matmul(x_shift, embedding_layer_out)
            x_out = x_out + self.bias_out
        else:
            x_out = self.embedding_layer_out(x_shift)

        # encoder
        output = self.transformer(
            x_in,
            x_out,
            src_mask=None,
            tgt_mask=None,
            memory_mask=None,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=None
        )

        # decoder
        output = self.decoder(output)

        return output


class FinetuneModel(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=1,
        n_enc=6,
        n_dec=6,
        nhid=100,
        nlayers=6,
        dropout=0.2,
        activation="relu",
        ignore_padding=False,
        weight_constraint=False,
        do_pos=False,
        NN_params: dict = None,
        output_size=1,
        run_id=None,
    ):
        super(FinetuneModel, self).__init__()
        # variables
        NUM_LEN, CAT_LEN = EHRDataset.num_feature
        self.input_size = (NUM_LEN + CAT_LEN) * 2
        self.impute_method = EHRDataset.impute_method
        self.d_model = d_model
        self.ignore_padding = ignore_padding
        self.weight_constraint = weight_constraint
        self.run_id = run_id

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

        # Transformer
        self.transformer = TransformerModel(
            d_model=d_model,
            nhead=nhead,
            n_enc=n_enc,
            n_dec=n_dec,
            nhid=nhid,
            dropout=dropout,
            activation=activation,
            ignore_padding=ignore_padding,
            weight_constraint=weight_constraint,
        )

        # encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=nhid, dropout=dropout,
            activation=activation, batch_first=True
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layers, nlayers, encoder_norm)

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
            nn.GELU(),
            nn.Linear(hidden_size, output_size)
        )
        self.decoder.apply(weights_init)

    def set_transformer(self, transformer):
        self.transformer.load_state_dict(transformer.state_dict())

    def forward(self, Xs):
        x_num, x_cat, x_num_mask, x_cat_mask = Xs
        x = torch.cat([x_num, x_cat], 2)
        x_mask = torch.cat([x_num_mask, x_cat_mask], 2)
        x_mask_invert = (x_mask < 1.0).float()

        # pretrain output impute finetune missing part
        x_tilde = self.transformer(Xs)
        x_hat = x * x_mask + x_tilde * x_mask_invert
        x = torch.cat([x_hat, x_mask], 2)

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

        # encoder
        output = self.encoder(output, src_key_padding_mask=src_key_padding_mask)
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
