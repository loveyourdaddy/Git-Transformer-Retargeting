import json
import torch
from torch import optim
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from wandb import set_trace 
import option_parser
from datasets import get_character_names, create_dataset
from models import create_model
from models.base_model import BaseModel
from model import *

""" Attentnion Model """
# function of Q, K, V
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_head):        
        super().__init__()
        #d_head (64) : dim of key vector 
        self.scale = 1 / (d_head ** 0.5)

    def forward(self, Q, K, V, attn_mask):
        # Q,K,V: (bs, n_head, window, DoF), attn_mask: (bs, n_head, window, window)

        # (bs, n_head, window, window)
        scores = torch.matmul(Q, K.transpose(-1, -2)).mul_(self.scale)
        scores.masked_fill_(attn_mask, -1e9)

        # Softmax on last dim 
        attn_prob = nn.Softmax(dim = -1)(scores)
        
        context = torch.matmul(attn_prob, V)

        # context:(bs, n_head, window, DoF) attn_prob (bs, n_head, window, window)
        return context, attn_prob

class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        # animation parameters
        self.DoF = args.DoF
        # self.window_size = args.window_size
        # head parameters
        self.d_head = args.d_head
        self.n_head = args.n_head
        # hidden vector dim
        self.d_hidn = args.d_hidn
        
        """ Q, K, V Network : 전체 프레임을 한번에 읽고 attention을 찾음 """
        self.W_Q = nn.Linear(self.DoF, self.n_head * self.d_head) # W: (DoF, 256)
        self.W_K = nn.Linear(self.DoF, self.n_head * self.d_head)
        self.W_V = nn.Linear(self.DoF, self.n_head * self.d_head)
        # self.W_Q = nn.Conv1d(self.DoF, self.n_head * self.d_head, kernel_size=1, padding=0) # W: (91, 256)
        # self.W_K = nn.Conv1d(self.DoF, self.n_head * self.d_head, kernel_size=1, padding=0)
        # self.W_V = nn.Conv1d(self.DoF, self.n_head * self.d_head, kernel_size=1, padding=0)

        # Get attention value
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head)
        self.linear = nn.Linear(self.n_head * self.d_head, self.DoF)

    def forward(self, Q, K, V, attn_mask):
        # Q,K,V:(bs, window, DoF) attn_mask:(bs, window, window)
        batch_size = Q.size(0)

        """ Data Encoding 1 """
        # (bs, *DoF, window) -> (bs, *n_head*d_head, window) -> (bs, window, *n_head, *d_head) -> (bs, *n_head, window, *d_head) 
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)

        # head 갯수만큼 차원 추가 및 복사해두기 : (bs, window, window) -> (bs, n_head, window, window)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)
        
        # Attentinon 계산
        # context: (bs, n_head, window, d_head)
        context, attn_prob = self.scaled_dot_attn(q_s, k_s, v_s, attn_mask)

        # (bs, n_head, window, d_head) -> (bs, window, n_head * d_head)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_head)        

        # (bs,window,nhead*dhead) -> (bs, window, DoF)
        output = self.linear(context)

        return output, attn_prob, context

""" Feed Forward """
class PositionFeedForwardNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.DoF = args.DoF
        # self.window_size = args.window_size

        # Layer TODO: change to FeedForward
        # self.conv1 = nn.Conv1d(in_channels = self.window_size, out_channels = self.window_size * 4, kernel_size=1, padding=0)
        # self.conv2 = nn.Conv1d(in_channels = self.window_size * 4, out_channels = self.window_size, kernel_size=1, padding=0)

        self.linear1 = nn.Linear(in_features = self.DoF,     out_features = self.DoF * 4)
        self.linear2 = nn.Linear(in_features = self.DoF * 4, out_features = self.DoF) # 1프레임마다 1개의 feature을 추출합니다. 
        self.active = F.gelu
    
    def forward(self, inputs):
        # (bs, window, DoF)

        # window 차원에 conv 연산
        output = self.active(self.linear1(inputs)) 
        output = self.linear2(output)

        return output

""" Layers"""
class EncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        # animation parameters
        self.args = args
        self.DoF = args.DoF
        # self.window_size = args.window_size
        self.layer_norm_epsilon = args.layer_norm_epsilon

        # Layers
        self.self_attn = MultiHeadAttention(self.args)
        self.layer_norm1 = nn.LayerNorm(self.DoF, eps=self.layer_norm_epsilon)
        self.pos_ffn = PositionFeedForwardNet(self.args)
        self.layer_norm2 = nn.LayerNorm(self.DoF, eps=self.layer_norm_epsilon)

    def forward(self, inputs, attn_mask):
        # input & 아래 전부 (bs, window, DoF)

        # attention 
        att_outputs, attn_prob, context = self.self_attn(inputs, inputs, inputs, attn_mask)

        # residual in encoder & 마지막 차원에 대해 layer normalization
        att_outputs = self.layer_norm1(inputs + att_outputs)

        # ff
        ffn_outputs = self.pos_ffn(att_outputs)
        # residual  
        ffn_outputs = self.layer_norm2(ffn_outputs + att_outputs)

        return ffn_outputs, attn_prob, context
    
class DecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.DoF = args.DoF
        # self.window_size = args.window_size

        self.self_attn = MultiHeadAttention(self.args)
        self.layer_norm1 = nn.LayerNorm(self.DoF, eps=self.args.layer_norm_epsilon)
        self.dec_enc_attn = MultiHeadAttention(self.args)
        self.layer_norm2 = nn.LayerNorm(self.DoF, eps=self.args.layer_norm_epsilon)
        self.pos_ffn = PositionFeedForwardNet(self.args)
        self.layer_norm3 = nn.LayerNorm(self.DoF, eps=self.args.layer_norm_epsilon)
    
    def forward(self, dec_inputs, enc_outputs, self_attn_mask, dec_enc_attn_mask):
        
        # dec_inputs에 Self attention을 적용합니다. (bs, DoF, d_hidn), (bs, n_head, DoF, DoF)
        self_att_outputs, self_attn_prob, _ = self.self_attn(dec_inputs, dec_inputs, dec_inputs, self_attn_mask) #Q, K, V, attn
        self_att_outputs = self.layer_norm1(dec_inputs + self_att_outputs)
        
        # dec_enc attention을 적용합니다. (bs, window, DoF), (bs, n_head, window, window)
        # Q : self attn, K,V : enc_output, 
        dec_enc_att_outputs, dec_enc_attn_prob, _ = self.dec_enc_attn(self_att_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask) 
        dec_enc_att_outputs = self.layer_norm2(self_att_outputs + dec_enc_att_outputs)

        # (bs, window, DoF)
        ffn_outputs = self.pos_ffn(dec_enc_att_outputs)
        # residual 
        ffn_outputs = self.layer_norm3(dec_enc_att_outputs + ffn_outputs)

        # (bs, window, DoF)
        return ffn_outputs, self_attn_prob, dec_enc_attn_prob

""" sinusoial encoding of each sentence """ 
# n_seq: num of total seq(Sentence), d_hidn: 단어를 표시하는 벡터의 크기
def get_sinusoid_encoding_table(n_seq, d_hidn):
    # 포지션을 angle로 나타냄
    def cal_angle(position, i_hidn):
        return position / np.power(10000, 2 * (i_hidn // 2) / d_hidn)
    def get_posi_ang_vec(position):
        return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

    sinusoid_table = np.array([get_posi_ang_vec(i_seq) for i_seq in range(n_seq)])
    # (bs, posiiton value)
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    # (DoF+1 , 32): each 프레임(int)들을 32dim의 float으로 나타냅니다
    return sinusoid_table

def get_attn_pad_mask(seq_q, seq_k, i_pad):
    # seq: (bs, window, DoF)

    # seq k에서 0인 부분을 <pad>
    valueTensor = torch.bmm(seq_q, seq_k.transpose(1,2)) # batch matrix mult
    pad_attn_mask = valueTensor.data.eq(i_pad)

    # (bs,window,window)
    return pad_attn_mask

""" Encoder & Decoder """
class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.DoF = args.DoF
        # self.window_size = args.window_size

        """ Layer """
        self.layers = nn.ModuleList([EncoderLayer(self.args) for _ in range(self.args.n_layer)])
        
        # output(depth)의 갯수이니, 2차원으로 써줘야함 
        # self.fc1 = nn.Linear(self.window_size, self.window_size)
        self.conv1 = nn.Conv1d(self.DoF, self.DoF, kernel_size=1, padding=0)

    def forward(self, inputs):
        # (bs, length of frames, joints): (4, 91, 64)

        # (bs, DoF, window)
        outputs = self.conv1(inputs)
        # outputs = outputs + positions
        # outputs = self.fc1(outputs)

        """ Transpose for window """
        # (bs, DoF, window) -> (bs, window, DoF) (4,128,91)
        outputs = outputs.transpose(1,2)

        """ Get Pad """
        # (bs, n_head, window, window)
        attn_mask = get_attn_pad_mask(outputs, outputs, self.args.i_pad)
        
        """ 연산 """
        # get all attn_prob of all layer
        attn_probs = []
        for layer in self.layers:
            outputs, attn_prob, context = layer(outputs, attn_mask)
            attn_probs.append(attn_prob)
        
        """ Transpose for window """
        # (bs, DoF, window) -> (bs, window, DoF) (4,128,91)
        outputs = outputs.transpose(1,2)
        
        return outputs, attn_probs, context

""" attention decoder mask: 현재단어와 이전단어는 볼 수 있고 다음단어는 볼 수 없도록 Masking 합니다. """
def get_attn_decoder_mask(seq): 

    # for conv2d 
    # seq = torch.squeeze(seq, 0)    

    seq_tensor = torch.matmul(seq, seq.transpose(1,2))
    subsequent_mask = seq_tensor.triu(diagonal=1) 

    # subsequent_mask = torch.unsqueeze(subsequent_mask, 0)
    
    return subsequent_mask

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.DoF = args.DoF
        # self.window_size = args.window_size
        self.d_hidn = args.d_hidn

        # self.dec_input_emb = nn.Embedding(self.DoF, self.args.d_hidn)
        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(self.args.DoF + 1, self.args.d_hidn))
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)

        # layers
        self.layers = nn.ModuleList([DecoderLayer(self.args) for _ in range(self.args.n_layer)])

        # output(depth)의 갯수이니, 2차원으로 써줘야함 
        # self.fc1 = nn.Linear(self.window_size, self.window_size)
        self.conv1 = nn.Conv1d(self.DoF, self.DoF, kernel_size=1, padding=0) # 차원 뒤집은 다음에사용하기 

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """ 연산 """
        # (bs, DoF, d_hidn)
        dec_outputs = self.conv1(dec_inputs)
        # dec_outputs = dec_outputs + positions
        
        """ Transpose for window """
        # (bs, DoF, window) -> (bs, window, DoF) (4,128,91)
        dec_outputs = dec_outputs.transpose(1,2)
        
        # (bs, DoF, DoF)
        dec_attn_pad_mask = get_attn_pad_mask(dec_outputs, dec_outputs, self.args.i_pad)
        # (bs, DoF, DoF)
        dec_attn_decoder_mask = get_attn_decoder_mask(dec_outputs)

        # (32, 913, 913)
        dec_self_attn_mask = torch.gt((dec_attn_pad_mask + dec_attn_decoder_mask), 0)
        # (bs, DoF, DoF)
        enc_inputs = enc_inputs.transpose(1,2)
        enc_outputs = enc_outputs.transpose(1,2)
        dec_enc_attn_mask = get_attn_pad_mask(dec_outputs, enc_inputs, self.args.i_pad)

        self_attn_probs, dec_enc_attn_probs = [], []
        for layer in self.layers:
            dec_outputs, self_attn_prob, dec_enc_attn_prob = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)

            # 모든 layer의 attn 을 쌓기 
            self_attn_probs.append(self_attn_prob)
            dec_enc_attn_probs.append(dec_enc_attn_prob)

        """ Transpose for window """
        # (bs, window, DoF) ->  (bs, DoF, window) (4,91,128)
        dec_outputs = dec_outputs.transpose(1,2)

        # (bs, DoF, d_hidn), [(bs, DoF, DoF)], [(bs, DoF, DoF)]
        return dec_outputs, self_attn_probs, dec_enc_attn_probs

""" Transoformer Model """    
class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
    
    def forward(self, enc_inputs, dec_inputs):
        # input: (bs, window, DoF), output: (bs, window, DoF)
        enc_outputs, enc_self_attn_probs, context = self.encoder(enc_inputs)
        
        # input: (bs, window, DoF), output: (bs, window, DoF)
        dec_outputs, dec_self_attn_probs, dec_enc_attn_probs = self.decoder(dec_inputs, enc_inputs, enc_outputs)

        return dec_outputs, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs
        

class MotionGenerator(nn.Module):
    def __init__(self, args, character_names, dataset):

        """ Fully Connected Layer"""
        # self.fc1 = nn.Linear(self.args.DoF, self.args.d_hidn)
        # self.fc2 = nn.Linear(self.args.d_hidn, self.args.DoF)
        """ 1d Conv layer """
        # 91 -> 64 -> 91
        # self.conv1 = nn.Conv1d(self.args.DoF, self.args.d_hidn, 1, padding=0) # d_hidn 개의 (output) kernel이 존재, kerner_size=3 # 15, 7
        # self.active = F.gelu
        # self.conv2 = nn.Conv1d(self.args.d_hidn, self.args.DoF, 1, padding=0)

        """ 2d conv layer """
        # self.conv1 = nn.Conv2d(self.args.DoF, self.args.d_hidn, (3,3), padding=1) # input_channel, output_channel, kernel(filter)_size
        # self.active = F.gelu
        # self.conv2 = nn.Conv2d(self.args.d_hidn, self.args.DoF, (3,3), padding=1)

        """ 1d auto-encoder"""
        # 91 -> 64 -> 91
        # self.conv1 = nn.Conv1d(self.args.DoF, self.args.d_hidn, 3, padding=1) # d_hidn 개의 (output) kernel이 존재, kerner_size=3 # 15, 7
        # self.active = F.gelu
        # self.deconv1 = nn.ConvTranspose1d(self.args.d_hidn, self.args.DoF, 3, padding=1)

        """ 2d auto-encoder"""
        # self.conv1 = nn.Conv2d(1, 1, 3, padding=1) # d_hidn 개의 (output) kernel이 존재, kerner_size=3 # 15, 7
        # self.active = F.gelu
        # self.deconv1 = nn.ConvTranspose2d(1, 1, 3, padding=1)

        """ Transformer """
        # Parameters 
        super().__init__()
        self.args = args
        self.DoF = args.DoF
        # self.window_size = args.window_size

        # layers
        self.transformer = Transformer(args)
        # self.projection = nn.Linear(self.DoF, self.DoF)
        self.projection = nn.Conv1d(self.DoF, self.DoF, kernel_size=1, padding=0)
        # self.param = self.transformer.parameters() + self.projection.parameters()

    """ Fuclly Connected layer Forward"""
    # def forward(self, enc_inputs, dec_inputs):
    #     x = self.fc1(enc_inputs)
    #     x_act = nn.Softmax(dim = -1)(x)
    #     y = self.fc2(x_act)
    #     return y

    """ Convolutional network Forward """
    # def forward(self, enc_inputs, dec_inputs):
    #     x = self.conv1(enc_inputs) 
    #     x_pool = self.active(x)
    #     y = self.conv2(x_pool)
    #     # print(y.size())
    #     # print(enc_inputs.size())
    #     return y

    """ auto-encoder Forward """
    # def forward(self, enc_inputs, dec_inputs):
    #     enc_inputs = torch.unsqueeze(enc_inputs,1)
    #     x = self.conv1(enc_inputs)
    #     x_act = self.active(x)
    #     y = self.deconv1(x_act)
    #     y = torch.squeeze(y,1)
    #     return y
        
    """ Transofrmer """
    def forward(self, enc_inputs, dec_inputs):
        dec_outputs, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs= self.transformer(enc_inputs, dec_inputs)
        
        output = self.projection(dec_outputs)

        return output, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs

    # def parameters(self, recurse: bool) -> Iterator[Parameter]:
    #     return super().parameters(recurse=recurse)
