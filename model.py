import json
import torch
from torch import optim
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
# from wandb import set_trace
from datasets import get_character_names, create_dataset
from model import *
import math
from models.Kinematics import ForwardKinematics
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
from torch.nn import LayerNorm


class MotionGenerator(nn.Module):
    def __init__(self, args, offsets, i):
        super().__init__()

        """ Transformer """
        self.transformer = Transformer(args, offsets, i).to(args.cuda_device)
        # self.discriminator = Discriminator(args, offsets, i).to(args.cuda_device)

    """ Transofrmer """

    def forward(self, src, tgt):

        return

    def G_parameters(self):
        return list(self.transformer.parameters())

    # def D_parameters(self):
    #     return list(self.discriminator.parameters())


class Discriminator(nn.Module):
    def __init__(self, args, offsets, i):
        super().__init__()
        self.transformer = Transformer(args, offsets, i).to(args.cuda_device)

    def forward(self, src, tgt):
        encoder_output = self.transformer.enc_forward(src)
        output = self.transformer.dec_forward(encoder_output, tgt, src)

        # TODO: one of sigmoid is enough??? no fc?
        return torch.sigmoid(output).squeeze()


""" Transoformer Model """


class Transformer(nn.Module):
    def __init__(self, args, offsets, i):
        super().__init__()

        if i == 0:
            self.input_dim = args.input_size
            self.output_dim = args.input_size
        else:
            self.input_dim = args.output_size
            self.output_dim = args.output_size

        self.hidden_dim = args.d_hidn  # embedding dimension
        self.num_heads = args.n_head
        self.num_layers = args.n_layer
        dropout = 0.5

        self.encoder = nn.Linear(self.input_dim, self.input_dim)
        self.project = nn.Linear(self.input_dim, self.input_dim)

    def enc_forward(self, src):
        for i in range(self.num_layers):
            src = self.encoder(src)
        return src

    def dec_forward(self, encoder_output, tgt, src):

        for i in range(self.num_layers):
            encoder_output = self.project(encoder_output)

        return encoder_output

    def forward(self, src, tgt):
        encoder_output = self.enc_forward(src)
        output = self.dec_forward(encoder_output, tgt, src)

        return output, encoder_output

# class Transformer(nn.Module):
#     def __init__(self, args, offsets, i):
#         super().__init__()
#         # self.args = args
#         # self.input_dim = args.window_size
#         # self.output_dim = args.window_size

#         if i == 0:
#             self.input_dim = args.input_size
#             self.output_dim = args.input_size
#         else:
#             self.input_dim = args.output_size
#             self.output_dim = args.output_size

#         self.hidden_dim = args.d_hidn  # embedding dimension
#         self.num_heads = args.n_head
#         self.num_layers = args.n_layer
#         dropout = 0.5

#         self.encoder = nn.Linear(self.input_dim, self.hidden_dim)
#         self.pos_encoder = PositionalEncoding(self.hidden_dim, dropout)
#         self.project = nn.Linear(self.hidden_dim, self.input_dim)

#         self.transformer_encoder = Transformer_Encoder(
#             self.hidden_dim, self.num_heads, self.num_layers, dropout)
#         self.transformer_decoder = Transformer_Decoder(
#             self.hidden_dim, self.num_heads, self.num_layers, dropout)

#     def _generate_square_subsequent_mask(self, sz):
#         mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#         mask = (
#             mask.float()
#             .masked_fill(mask == 0, float("-inf"))
#             .masked_fill(mask == 1, float(0.0))
#         )
#         return mask

    # def enc_forward(self, src):
    #     projected_src = self.encoder(src) * np.sqrt(self.input_dim)
    #     pos_encoded_src = self.pos_encoder(projected_src)
    #     encoder_output = self.transformer_encoder(pos_encoded_src)
    #     return encoder_output

    # def dec_forward(self, encoder_output, tgt, src):
    #     # mask
    #     tgt_mask = self._generate_square_subsequent_mask(tgt.shape[0]).to(
    #         device=tgt.device,
    #     )

    #     # Use last source pose as first input to decoder
    #     # tgt = torch.cat((src[-1].unsqueeze(0), tgt[:-1]))
    #     pos_encoder_tgt = self.pos_encoder(
    #         self.encoder(tgt) * np.sqrt(self.input_dim)
    #     )
    #     output = self.transformer_decoder(
    #         encoder_output, pos_encoder_tgt, tgt_mask)
    #     output = self.project(output)

    #     return output

    # def forward(self, src, tgt):
    #     encoder_output = self.enc_forward(src)
    #     output = self.dec_forward(encoder_output, tgt, src)

    #     # , enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs
    #     return output, encoder_output


class Transformer_Encoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        encoder_layer = TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dropout=dropout
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.num_layers,
            norm=LayerNorm(self.hidden_dim),
        )

    def forward(self, pos_encoded_src):
        encoder_output = self.transformer_encoder(pos_encoded_src)

        return encoder_output


class Transformer_Decoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        decoder_layer = TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dropout=dropout
        )
        self.transformer_decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=self.num_layers,
            norm=LayerNorm(self.hidden_dim),
        )

    def forward(self, encoder_output, pos_encoder_tgt, tgt_mask):
        output = self.transformer_decoder(
            pos_encoder_tgt, encoder_output, tgt_mask=tgt_mask,
        )
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.5, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
