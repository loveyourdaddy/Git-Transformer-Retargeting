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
# from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
# from torch.nn import LayerNorm

class MotionGenerator(nn.Module):
    def __init__(self, args, offsets, i):
        super().__init__()
        self.args = args
        self.output_size = args.window_size
        # if args.swap_dim == 0:
        #     self.output_size = args.output_size
        # else:
        self.output_size = args.window_size

        """ Transformer """
        # layers
        self.transformer = Transformer(args, offsets, i).to(args.cuda_device)

    """ Transofrmer """
    def forward(self, enc_inputs, gt):
        # output #, latent_feature, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs = self.transformer(enc_inputs, gt)

        return enc_inputs # output #, latent_feature #, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs

    def G_parameters(self):
        return list(self.transformer.parameters())

""" Transoformer Model """
class Transformer(nn.Module):
    def __init__(self, args, offsets, i):
        super().__init__()
        # self.args = args
        self.input_dim = args.window_size
        self.output_dim = args.window_size
        self.hidden_dim = args.d_hidn # embedding dimension 
        self.num_heads = args.n_head
        self.num_layers = args.n_layer
        # self.d_model = 512

        dropout = 0.5

        self.encoder = nn.Linear(self.input_dim, self.hidden_dim).to(args.cuda_device) # ntoken, ninp
        self.project = nn.Linear(self.hidden_dim, self.output_dim) # ninp, ntoken

        # self.pos_encoder = PositionalEncoding(self.hidden_dim, dropout)

        # encoder_layer = TransformerEncoderLayer(
        #     d_model = self.hidden_dim,
        #     nhead = self.num_heads,
        #     # dim_feedforward = self.hidden_dim,
        #     dropout = dropout
        # )
        # self.transformer_encoder = TransformerEncoder(
        #     encoder_layer=encoder_layer,
        #     num_layers=self.num_layers,
        #     norm=LayerNorm(self.input_dim),
        # )

        # decoder_layer = TransformerDecoderLayer(
        #     d_model = self.hidden_dim,
        #     nhead = self.num_heads,
        #     # dim_feedforward = self.hidden_dim,
        #     dropout = dropout
        # )
        # self.transformer_decoder = TransformerDecoder(
        #     decoder_layer=decoder_layer,
        #     num_layers=self.num_layers,
        #     norm=LayerNorm(self.output_dim),
        # )

    # def _generate_square_subsequent_mask(self, sz):
    #     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    #     mask = (
    #         mask.float()
    #         .masked_fill(mask == 0, float("-inf"))
    #         .masked_fill(mask == 1, float(0.0))
    #     )
    #     return mask
        
    def forward(self, src, tgt):
        # enc 
        # self.encoder = nn.Linear(self.input_dim, self.hidden_dim) # ntoken, ninp
        tmp_net = nn.Linear(self.input_dim, self.hidden_dim).to('cuda:0') # ntoken, ninp
        tmp = torch.rand(src.shape).to('cuda:0')
        
        projected_src = self.encoder(src) * np.sqrt(self.input_dim)
        pos_encoded_src = self.pos_encoder(projected_src)
        encoder_output = self.transformer_encoder(pos_encoded_src)

        # dec
        tgt_mask = self._generate_square_subsequent_mask(tgt.shape[0]).to(
            device=tgt.device,
        )

        # Use last source pose as first input to decoder
        tgt = torch.cat((src[-1].unsqueeze(0), tgt[:-1]))
        pos_encoder_tgt = self.pos_encoder(
            self.encoder(tgt) * np.sqrt(self.ninp)
        )
        output = self.transformer_decoder(
            pos_encoder_tgt, encoder_output, tgt_mask=tgt_mask,
        )
        output = self.project(output)

        return output #, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs

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

# """ attention pad mask """
# # TODO: check here
# def get_attn_pad_mask(seq_q, seq_k, i_pad):
#     batch_size, len_q = seq_q.size()
#     batch_size, len_k = seq_k.size()
#     pad_attn_mask = seq_k.data.eq(i_pad).unsqueeze(1).expand(batch_size, len_q, len_k)  # <pad>
#     return pad_attn_mask

# """ attention decoder mask """
# def get_attn_decoder_mask(seq):
#     subsequent_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
#     subsequent_mask = subsequent_mask.triu(diagonal=1) # upper triangular part of a matrix(2-D)
#     return subsequent_mask

# """ Encoder & Decoder """
# class Encoder(nn.Module):
#     def __init__(self, args, offset, i):
#         super().__init__()
#         self.args = args
#         self.offset = offset
#         self.window_size = args.window_size
#         self.embedding_dim = args.embedding_dim
#         # if args.swap_dim == 0:
#         #     self.input_size = args.input_size
#         # else:
#         #     self.input_size = args.window_size
#         # if i == 0:
#         #     self.input_size = args.input_size
#         # else:
#         #     self.input_size = args.output_size

#         """ Embedding networks """
#         # input embedding
#         self.input_embedding = nn.Linear(self.window_size, self.embedding_dim)
#         # Positional Embedding
#         self.sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(self.window_size + 1, self.embedding_dim))
#         self.pos_emb = nn.Embedding.from_pretrained(self.sinusoid_table, freeze=True)

#         """ Layer """
#         self.fc1 = nn.Linear(self.embedding_dim, self.embedding_dim)
#         self.layers = nn.ModuleList([EncoderLayer(self.args) for _ in range(self.args.n_layer)])
#         self.fc2 = nn.Linear(self.embedding_dim, self.embedding_dim)
#         self.projection_net = ProjectionNet(args, i)

#     def forward(self, inputs):
#         """ Get Position and Embedding """
#         data_encoding = 1
#         if data_encoding:
#             positions = torch.arange(inputs.size(1), device=inputs.device, dtype=torch.long)\
#                 .unsqueeze(0).expand(inputs.size(0), inputs.size(1)).contiguous() + 1
#             position_encoding = self.pos_emb(positions)
#             input_embedding = self.input_embedding(inputs)
#             inputs = input_embedding + position_encoding

#         outputs = self.fc1(inputs)

#         attn_mask = get_attn_pad_mask(outputs, outputs, 0)

#         """ 연산 """
#         attn_probs = []
#         for layer in self.layers:
#             outputs, attn_prob, context = layer(outputs, attn_mask)
#             attn_probs.append(attn_prob)

#         outputs = self.fc2(outputs)

#         latent_feature = self.projection_net(outputs)
 
#         return latent_feature, attn_probs, context

# class Decoder(nn.Module):
#     def __init__(self, args, offset, i):
#         super().__init__()
#         self.args = args
#         self.offset = offset
#         self.window_size = args.window_size
#         self.embedding_dim = args.embedding_dim
#         # if args.swap_dim == 0:
#         #     self.output_size = args.output_size
#         # else:
#         self.output_size = args.window_size

#         """ layers """
#         self.fc1 = nn.Linear(self.embedding_dim, self.embedding_dim)
#         self.layers = nn.ModuleList([DecoderLayer(self.args) for _ in range(self.args.n_layer)])
#         self.fc2 = nn.Linear(self.embedding_dim, self.embedding_dim)
#         self.de_embedding = nn.Linear(self.embedding_dim, self.output_size)
#         self.deprojection_net = DeprojectionNet(args, i)

#         """ Embedding networks """
#         # input embedding
#         self.input_embedding = nn.Linear(self.window_size, self.embedding_dim)
#         # Positional Embedding
#         self.sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(self.window_size + 1, self.embedding_dim))
#         self.pos_emb = nn.Embedding.from_pretrained(self.sinusoid_table, freeze=True)
           
#     def forward(self, dec_inputs, enc_outputs):
#         # 1. enc output
#         enc_outputs = self.deprojection_net(enc_outputs)
#         enc_outputs = self.fc1(enc_outputs)
        
#         # 2. dec input
#         """ Get Position and Embedding """
#         data_encoding = 1
#         if data_encoding:
#             positions = torch.arange(dec_inputs.size(1), device=dec_inputs.device, dtype=torch.long)\
#                 .unsqueeze(0).expand(dec_inputs.size(0), dec_inputs.size(1)).contiguous() + 1
#             position_encoding = self.pos_emb(positions)
#             input_embedding = self.input_embedding(dec_inputs)
#             dec_inputs = input_embedding + position_encoding

#         dec_inputs = self.fc1(dec_inputs)

#         """ Decoder mask """
#         dec_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, 0)
#         dec_attn_decoder_mask = get_attn_decoder_mask(dec_inputs) # To know the rule
#         dec_self_attn_mask = torch.gt((dec_attn_pad_mask + dec_attn_decoder_mask), 0)

#         dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs, 0) # ? enc input? 

#         self_attn_probs, dec_enc_attn_probs = [], []
#         for layer in self.layers:
#             outputs, self_attn_prob, dec_enc_attn_prob = layer(dec_inputs, enc_outputs) # Q, (K,V)
#             self_attn_probs.append(self_attn_prob)
#             dec_enc_attn_probs.append(dec_enc_attn_prob)
 
#         outputs = self.fc1(outputs)

#         outputs = self.de_embedding(outputs)

#         return outputs, self_attn_probs, dec_enc_attn_probs

# """ Layers """
# class EncoderLayer(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         # animation parameters
#         self.args = args
#         self.input_dim = args.embedding_dim  # args.input_size
#         self.layer_norm_epsilon = args.layer_norm_epsilon

#         # Layers
#         self.self_attn = MultiHeadAttention(self.args, "Enc")  # Q,K,V: (bs, 128, 91)
#         self.layer_norm1 = nn.LayerNorm(self.input_dim, eps=self.layer_norm_epsilon)
#         self.pos_ffn = PositionFeedForwardNet(self.args, "Enc")
#         self.layer_norm2 = nn.LayerNorm(self.input_dim, eps=self.layer_norm_epsilon)

#     def forward(self, inputs, mask):
#         att_outputs, attn_prob, context = self.self_attn(inputs, inputs, inputs, mask)
#         att_outputs = self.layer_norm1(inputs + att_outputs)

#         ffn_outputs = self.pos_ffn(att_outputs)
#         ffn_outputs = self.layer_norm2(ffn_outputs + att_outputs)

#         return ffn_outputs, attn_prob, context


# class DecoderLayer(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.input_dim = args.embedding_dim

#         self.self_attn = MultiHeadAttention(self.args, "Dec")  # Q,K,V: (bs, 128, 111)
#         self.layer_norm1 = nn.LayerNorm(self.input_dim, eps=self.args.layer_norm_epsilon) # Q: (bs, 128, 111), K,V: (bs, 128, 91)
#         self.dec_enc_attn = MultiHeadAttention(self.args, "Dec_enc")
#         self.layer_norm2 = nn.LayerNorm(self.input_dim, eps=self.args.layer_norm_epsilon)
#         self.pos_ffn = PositionFeedForwardNet(self.args, "Dec")
#         self.layer_norm3 = nn.LayerNorm(self.input_dim, eps=self.args.layer_norm_epsilon)

#     def forward(self, dec_inputs, enc_outputs):

#         # decoder self attention 
#         self_att_outputs, self_attn_prob, _ = self.self_attn(dec_inputs, dec_inputs, dec_inputs)  # Q, K, V, attn
#         self_att_outputs = self.layer_norm1(dec_inputs + self_att_outputs)

#         # source-target attentino  
#         dec_enc_att_outputs, dec_enc_attn_prob, _ = self.dec_enc_attn(self_att_outputs, enc_outputs, enc_outputs) # Q: output of decoder. K, V: output of encoder 
#         dec_enc_att_outputs = self.layer_norm2(self_att_outputs + dec_enc_att_outputs)

#         ffn_outputs = self.pos_ffn(dec_enc_att_outputs)
#         ffn_outputs = self.layer_norm3(dec_enc_att_outputs + ffn_outputs)

#         return ffn_outputs, self_attn_prob, dec_enc_attn_prob

# """ sinusoial encoding of each sentence """
# # n_seq: num of total seq(Sentence), d_hidn: 단어를 표시하는 벡터의 크기
# def get_sinusoid_encoding_table(n_seq, d_hidn):
#     def cal_angle(position, i_hidn):     # 포지션을 angle로 나타냄
#         return position / np.power(10000, 2 * (i_hidn // 2) / d_hidn)

#     def get_posi_ang_vec(position):
#         return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

#     sinusoid_table = np.array([get_posi_ang_vec(i_seq) for i_seq in range(n_seq)])

#     sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
#     sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

#     return sinusoid_table

# class MultiHeadAttention(nn.Module):
#     def __init__(self, args, type):
#         super().__init__()

#         self.input_dim, self.Q_input_dim, self.K_input_dim, self.V_input_dim = args.embedding_dim, args.embedding_dim, args.embedding_dim, args.embedding_dim
   
#         # head parameters
#         self.d_head = args.d_head
#         self.n_head = args.n_head
#         # hidden vector dim
#         self.d_hidn = args.d_hidn

#         """ Q, K, V Network : 전체 프레임을 한번에 읽고 attention을 찾음 """
#         self.W_Q = nn.Linear(self.Q_input_dim, self.n_head * self.d_head)
#         self.W_K = nn.Linear(self.K_input_dim, self.n_head * self.d_head)
#         self.W_V = nn.Linear(self.V_input_dim, self.n_head * self.d_head)

#         # Get attention value
#         self.scaled_dot_attn = ScaledDotProductAttention(args)
#         self.linear = nn.Linear(self.n_head * self.d_head, self.input_dim)

#     def forward(self, Q, K, V, mask):
#         # Q,K,V:(bs, window, DoF)
#         batch_size = Q.size(0)

#         """ Data Encoding 1 """
#         q_s = self.W_Q(Q).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
#         k_s = self.W_K(K).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
#         v_s = self.W_V(V).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)

#         # TODO: check dimension 
#         mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

#         # Attentinon 계산
#         context, attn_prob = self.scaled_dot_attn(q_s, k_s, v_s, mask)

#         context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_head)
        
#         # (bs,window,nhead*dhead) -> (bs, window, DoF)
#         output = self.linear(context)

#         return output, attn_prob, context

# """ Attentnion Model """
# class ScaledDotProductAttention(nn.Module):
#     def __init__(self, args):
#         super().__init__()

#         self.d_head = args.d_head
#         self.scale = 1 / (self.d_head ** 0.5)

#     def forward(self, Q, K, V):

#         scores = torch.matmul(Q, K.transpose(-1, -2)).mul_(self.scale)

#         attn_prob = nn.Softmax(dim=-1)(scores)

#         context = torch.matmul(attn_prob, V)

#         return context, attn_prob

# """ Feed Forward """
# class PositionFeedForwardNet(nn.Module):
#     def __init__(self, args, type):
#         super().__init__()
#         if type == "Enc":
#             self.input_dim = args.embedding_dim  # args.input_size
#         elif type == "Dec":
#             self.input_dim = args.embedding_dim  # args.output_size
#         else:  # EncDec
#             print("error")

#         self.linear1 = nn.Linear(
#             in_features=self.input_dim, out_features=self.input_dim * 2)
#         # 1프레임마다 1개의 feature을 추출합니다.
#         self.linear2 = nn.Linear(
#             in_features=self.input_dim * 2, out_features=self.input_dim)
#         self.active = F.gelu

#     def forward(self, inputs):
#         # (bs, window, DoF)

#         output = self.active(self.linear1(inputs))
#         output = self.linear2(output)

#         return output


""" projection and deprojection """
class ProjectionNet(nn.Module):
    def __init__(self, args, i):
        super().__init__()
        self.args = args
        # if i == 0:
        #     self.input_dim = args.input_size
        # else: 
        #     self.input_dim = args.output_size        
        if i == 0:
            self.input_dim = args.input_size
        else:
            self.input_dim = args.output_size
        self.d_hidn = args.d_hidn 

        """ layer """
        self.fc1 = nn.Linear(self.input_dim, self.d_hidn)

    def forward(self, enc_output):
        enc_output = torch.transpose(enc_output, 1, 2)
        # (bs, dimenedding, input DoF) -> (bs, dimenedding, hidn) 
        latent_img = self.fc1(enc_output)

        return latent_img

class DeprojectionNet(nn.Module):
    def __init__(self, args, i):
        super().__init__()
        self.args = args
        self.d_hidn = args.d_hidn
        # self.output_dim = args.output_size

        if i == 0:
            self.output_dim = args.input_size
        else: 
            self.output_dim = args.output_size

        """ layer """
        self.fc2 = nn.Linear(self.d_hidn, self.output_dim)

    def forward(self, latent_img):
        # (bs, dimenedding, input DoF) -> (bs, dimenedding, hidn) 
        dec_input = self.fc2(latent_img)

        dec_input = torch.transpose(dec_input, 1, 2)

        return dec_input
