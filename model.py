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

SAVE_ATTENTION_DIR = "attention_vis/test"

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

#     def forward(self, Q, K, V):
#         # Q,K,V:(bs, window, DoF)
#         batch_size = Q.size(0)

#         """ Data Encoding 1 """
#         q_s = self.W_Q(Q).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
#         k_s = self.W_K(K).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
#         v_s = self.W_V(V).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)

#         # Attentinon 계산
#         context, attn_prob = self.scaled_dot_attn(q_s, k_s, v_s)

#         context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_head)
        
#         # (bs,window,nhead*dhead) -> (bs, window, DoF)
#         output = self.linear(context)

#         return output, attn_prob, context


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
#             in_features=self.input_dim,     out_features=self.input_dim * 4)
#         # 1프레임마다 1개의 feature을 추출합니다.
#         self.linear2 = nn.Linear(
#             in_features=self.input_dim * 4, out_features=self.input_dim)
#         self.active = F.gelu

#     def forward(self, inputs):
#         # (bs, window, DoF)

#         output = self.active(self.linear1(inputs))
#         output = self.linear2(output)

#         return output


# """ Layers """
# class EncoderLayer(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         # animation parameters
#         self.args = args
#         self.input_dim = args.embedding_dim  # args.input_size
#         self.layer_norm_epsilon = args.layer_norm_epsilon

#         # Layers
#         self.self_attn = MultiHeadAttention(
#             self.args, "Enc")  # Q,K,V: (bs, 128, 91)
#         self.layer_norm1 = nn.LayerNorm(
#             self.input_dim, eps=self.layer_norm_epsilon)
#         self.pos_ffn = PositionFeedForwardNet(self.args, "Enc")
#         self.layer_norm2 = nn.LayerNorm(
#             self.input_dim, eps=self.layer_norm_epsilon)

#     def forward(self, inputs):
#         att_outputs, attn_prob, context = self.self_attn(
#             inputs, inputs, inputs)
#         att_outputs = self.layer_norm1(inputs + att_outputs)

#         ffn_outputs = self.pos_ffn(att_outputs)
#         ffn_outputs = self.layer_norm2(ffn_outputs + att_outputs)

#         return ffn_outputs, attn_prob, context


# class DecoderLayer(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.input_dim = args.embedding_dim

#         self.self_attn = MultiHeadAttention(
#             self.args, "Dec")  # Q,K,V: (bs, 128, 111)
#         self.layer_norm1 = nn.LayerNorm(
#             self.input_dim, eps=self.args.layer_norm_epsilon)
#         # Q: (bs, 128, 111), K,V: (bs, 128, 91)
#         self.dec_enc_attn = MultiHeadAttention(self.args, "Dec_enc")
#         self.layer_norm2 = nn.LayerNorm(
#             self.input_dim, eps=self.args.layer_norm_epsilon)
#         self.pos_ffn = PositionFeedForwardNet(self.args, "Dec")
#         self.layer_norm3 = nn.LayerNorm(
#             self.input_dim, eps=self.args.layer_norm_epsilon)

#     def forward(self, dec_inputs, enc_outputs):

#         self_att_outputs, self_attn_prob, _ = self.self_attn(
#             dec_inputs, dec_inputs, dec_inputs)  # Q, K, V, attn
#         self_att_outputs = self.layer_norm1(dec_inputs + self_att_outputs)

#         dec_enc_att_outputs, dec_enc_attn_prob, _ = self.dec_enc_attn(
#             self_att_outputs, enc_outputs, enc_outputs)
#         dec_enc_att_outputs = self.layer_norm2(
#             self_att_outputs + dec_enc_att_outputs)

#         ffn_outputs = self.pos_ffn(dec_enc_att_outputs)
#         ffn_outputs = self.layer_norm3(dec_enc_att_outputs + ffn_outputs)

#         return ffn_outputs, self_attn_prob, dec_enc_attn_prob


# """ sinusoial encoding of each sentence """
# # n_seq: num of total seq(Sentence), d_hidn: 단어를 표시하는 벡터의 크기
# def get_sinusoid_encoding_table(n_seq, d_hidn):
#     # 포지션을 angle로 나타냄
#     def cal_angle(position, i_hidn):
#         return position / np.power(10000, 2 * (i_hidn // 2) / d_hidn)

#     def get_posi_ang_vec(position):
#         return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

#     sinusoid_table = np.array([get_posi_ang_vec(i_seq) for i_seq in range(n_seq)])

#     sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
#     sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

#     return sinusoid_table

# class ProjectionNet(nn.Module):
#     def __init__(self, args, i):
#         super().__init__()
#         self.args = args
#         if i == 0:
#             self.input_dim = args.input_size
#         else: 
#             self.input_dim = args.output_size
#         self.d_hidn = args.d_hidn

#         """ layer """
#         self.fc1 = nn.Linear(self.input_dim, self.d_hidn)

#     def forward(self, enc_output):
#         enc_output = torch.transpose(enc_output, 1, 2)
#         # (bs, dimenedding, input DoF) -> (bs, dimenedding, hidn) 
#         latent_img = self.fc1(enc_output)

#         return latent_img

# class DeprojectionNet(nn.Module):
#     def __init__(self, args, i):
#         super().__init__()
#         self.args = args
#         self.d_hidn = args.d_hidn
#         # self.output_dim = args.output_size

#         if i == 0:
#             self.output_dim = args.input_size
#         else: 
#             self.output_dim = args.output_size

#         """ layer """
#         self.fc2 = nn.Linear(self.d_hidn, self.output_dim)

#     def forward(self, latent_img):
#         # (bs, dimenedding, input DoF) -> (bs, dimenedding, hidn) 
#         dec_input = self.fc2(latent_img)

#         dec_input = torch.transpose(dec_input, 1, 2)

#         return dec_input

# """ Encoder & Decoder """
# class Encoder(nn.Module):
#     def __init__(self, args, offset, i):
#         super().__init__()
#         self.args = args
#         self.offset = offset
#         self.window_size = args.window_size
#         self.embedding_dim = args.embedding_dim
#         if args.swap_dim == 0:
#             self.input_size = args.input_size
#         else:
#             self.input_size = args.window_size

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

#     def forward(self, input_character, inputs, data_encoding):
#         """ Get Position and Embedding """
#         if data_encoding:
#             positions = torch.arange(inputs.size(1), device=inputs.device, dtype=torch.long)\
#                 .unsqueeze(0).expand(inputs.size(0), inputs.size(1)).contiguous() + 1
#             position_encoding = self.pos_emb(positions)
#             input_embedding = self.input_embedding(inputs)
#             inputs = input_embedding + position_encoding

#         outputs = self.fc1(inputs)

#         """ 연산 """
#         attn_probs = []
#         for layer in self.layers:
#             outputs, attn_prob, context = layer(outputs)
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
#         if args.swap_dim == 0:
#             self.output_size = args.output_size
#         else:
#             self.output_size = args.window_size

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
            
#     def forward(self, output_character, inputs, enc_outputs, data_encoding): # inputs = dec_inputs 

#         # """ Get Position and Embedding """
#         # if data_encoding: # data_encoding for "inputs"
#         #     positions = torch.arange(inputs.size(1), device=inputs.device, dtype=torch.long)\
#         #         .unsqueeze(0).expand(inputs.size(0), inputs.size(1)).contiguous() + 1
#         #     position_encoding = self.pos_emb(positions)
#         #     input_embedding = self.input_embedding(inputs)
#         #     inputs = input_embedding + position_encoding

#         # 1. enc output
#         outputs = self.deprojection_net(enc_outputs)
        
#         outputs = self.fc1(outputs)

#         # 2. dec input
#         # dec_inputs = self.fc1(inputs)

#         self_attn_probs, dec_enc_attn_probs = [], []
#         for layer in self.layers:
#             outputs, self_attn_prob, dec_enc_attn_prob = layer(outputs, outputs)
#             self_attn_probs.append(self_attn_prob)
#             dec_enc_attn_probs.append(dec_enc_attn_prob)
 
#         outputs = self.fc1(outputs)

#         outputs = self.de_embedding(outputs)

#         return outputs, self_attn_probs, dec_enc_attn_probs

# """ Transoformer Model """
# class Transformer(nn.Module):
#     def __init__(self, args, offsets, i):
#         super().__init__()
#         self.args = args

#         # layers 
#         self.encoder = Encoder(args, offsets[i], i)
#         self.decoder = Decoder(args, offsets[i], i)

#     def forward(self, input_character, output_character, inputs):        
#         # Encoder
#         latent_feature, enc_self_attn_probs, context = self.encoder(input_character, inputs, data_encoding = 1)

#         outputs, dec_self_attn_probs, dec_enc_attn_probs = self.decoder(output_character, \
#             latent_feature, latent_feature, data_encoding = 1)

#         return outputs, self.decoder.de_embedding(self.decoder.deprojection_net(latent_feature)), enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs
#         # latent_feature

# class MotionGenerator(nn.Module):
#     def __init__(self, args, offsets, i):
#         super().__init__()
#         self.args = args
#         self.output_size = args.window_size
#         if args.swap_dim == 0:
#             self.output_size = args.output_size
#         else:
#             self.output_size = args.window_size

#         """ Transformer """
#         # layers
#         self.transformer = Transformer(args, offsets, i)

#     """ Transofrmer """
#     def forward(self, input_character, output_character, enc_inputs):
#         output, latent_feature, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs = self.transformer(
#             input_character, output_character, enc_inputs)

#         return output, latent_feature, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs

# """ Discriminator """
# class Discriminator(nn.Module):
#     def __init__(self, args, offsets, i):
#         super(Discriminator, self).__init__()
#         self.args = args
#         self.input_dim = args.window_size

#         """ layers """
#         self.transformer = Transformer(args, offsets, i)
#         # self.projection = nn.Linear(self.input_dim, self.input_dim)

#     def forward(self, input_character, output_character, enc_inputs):
#         output, _, _, _, _ = self.transformer(
#             input_character, output_character, enc_inputs)

#         # output = self.projection(output)

#         output = output.reshape(output.shape[0], -1)

#         return torch.sigmoid(output)

# """ Linear Encoder & Decoder """
# class Encoder(nn.Module):
#     def __init__(self, args, offset, i):
#         super().__init__()
#         self.input_dim = args.window_size
#         self.layers = nn.ModuleList([nn.Linear(self.input_dim, self.input_dim) for _ in range(args.n_layer)])

#     def forward(self, inputs):
#         outputs = inputs  
#         for layer in self.layers:
#             outputs = layer(inputs)
#         return outputs
        
# class Decoder(nn.Module):
#     def __init__(self, args, offset, i):
#         super().__init__()
#         self.input_dim = args.window_size
#         self.layers = nn.ModuleList([nn.Linear(self.input_dim, self.input_dim) for _ in range(args.n_layer)])

#     def forward(self, inputs):      
#         outputs = inputs  
#         for layer in self.layers:
#             outputs = layer(inputs)
#         return outputs    


""" Class for conv / pooling """
class SkeletonConv(nn.Module):
    def __init__(self, neighbour_list, in_channels, out_channels, kernel_size, joint_num, stride=1, padding=0,
                 bias=True, padding_mode='zeros', add_offset=False, in_offset_channel=0):
                 # padding_mode = reflection
        self.in_channels_per_joint = in_channels // joint_num
        self.out_channels_per_joint = out_channels // joint_num
        if in_channels % joint_num != 0 or out_channels % joint_num != 0:
            raise Exception('BAD')

        super(SkeletonConv, self).__init__()
        self.expanded_neighbour_list = []
        self._padding_repeated_twice = (padding, padding)
        self.padding_mode = padding_mode
        self.stride = stride
        self.dilation = 1
        self.groups = 1

        # expanded_neighbour_list: element index in vector, including channel(rotation) of joint (quaternion 4)
        for neighbour in neighbour_list:
            expanded = []
            for k in neighbour:
                for i in range(self.in_channels_per_joint):
                    expanded.append(k * self.in_channels_per_joint + i)
            self.expanded_neighbour_list.append(expanded)
        
        self.weight = torch.zeros(out_channels, in_channels, kernel_size)
        self.bias = torch.zeros(out_channels)
        self.mask = torch.zeros_like(self.weight)

        for i, neighbour in enumerate(self.expanded_neighbour_list):
            self.mask[self.out_channels_per_joint * i: self.out_channels_per_joint * (i + 1), neighbour, ...] = 1
        self.mask = nn.Parameter(self.mask, requires_grad=False) # nn parameter : not layer, only parameter (can learnable)

        self.description = 'SkeletonConv(in_channels_per_armature={}, out_channels_per_armature={}, kernel_size={}, ' \
                           'joint_num={}, stride={}, padding={}, bias={})'.format(
            in_channels // joint_num, out_channels // joint_num, kernel_size, joint_num, stride, padding, bias
        )

        self.reset_parameters()

    def reset_parameters(self):
        for i, neighbour in enumerate(self.expanded_neighbour_list):
            """ Use temporary variable to avoid assign to copy of slice, which might lead to un expected result """
            tmp = torch.zeros_like(self.weight[self.out_channels_per_joint * i: self.out_channels_per_joint * (i + 1),
                                   neighbour, ...])
            nn.init.kaiming_uniform_(tmp, a=math.sqrt(5))
            self.weight[self.out_channels_per_joint * i: self.out_channels_per_joint * (i + 1),
                        neighbour, ...] = tmp
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    self.weight[self.out_channels_per_joint * i: self.out_channels_per_joint * (i + 1), neighbour, ...])
                bound = 1 / math.sqrt(fan_in)
                tmp = torch.zeros_like(
                    self.bias[self.out_channels_per_joint * i: self.out_channels_per_joint * (i + 1)])
                nn.init.uniform_(tmp, -bound, bound)
                self.bias[self.out_channels_per_joint * i: self.out_channels_per_joint * (i + 1)] = tmp

        self.weight = nn.Parameter(self.weight)
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias)

    def set_offset(self, offset):
        if not self.add_offset: raise Exception('Wrong Combination of Parameters')
        self.offset = offset.reshape(offset.shape[0], -1)

    def forward(self, input):
        weight_masked = self.weight * self.mask
        res = F.conv1d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                       weight_masked, self.bias, self.stride,
                       0, self.dilation, self.groups)
        return res

class SkeletonPool(nn.Module):
    def __init__(self, edges, pooling_mode, channels_per_edge, last_pool=False):
        super(SkeletonPool, self).__init__()

        # check here 
        if pooling_mode != 'mean':
            raise Exception('Unimplemented pooling mode in matrix_implementation')

        self.channels_per_edge = channels_per_edge
        self.pooling_mode = pooling_mode
        self.edge_num = len(edges) + 1
        self.seq_list = []
        self.pooling_list = []
        self.new_edges = []
        degree = [0] * 100

        for edge in edges:
            degree[edge[0]] += 1
            degree[edge[1]] += 1

        def find_seq(j, seq):
            nonlocal self, degree, edges

            if degree[j] > 2 and j != 0:
                self.seq_list.append(seq)
                seq = []

            if degree[j] == 1:
                self.seq_list.append(seq)
                return

            for idx, edge in enumerate(edges):
                if edge[0] == j:
                    find_seq(edge[1], seq + [idx])

        find_seq(0, [])
        for seq in self.seq_list:
            if last_pool:
                self.pooling_list.append(seq)
                continue
            if len(seq) % 2 == 1:
                self.pooling_list.append([seq[0]])
                self.new_edges.append(edges[seq[0]])
                seq = seq[1:]
            for i in range(0, len(seq), 2):
                self.pooling_list.append([seq[i], seq[i + 1]])
                self.new_edges.append([edges[seq[i]][0], edges[seq[i + 1]][1]])

        # add global position
        self.pooling_list.append([self.edge_num - 1])

        self.description = 'SkeletonPool(in_edge_num={}, out_edge_num={})'.format(
            len(edges), len(self.pooling_list)
        )

        # why pooling need parameters? 
        self.weight = torch.zeros(len(self.pooling_list) * channels_per_edge, self.edge_num * channels_per_edge)

        for i, pair in enumerate(self.pooling_list):
            for j in pair:
                for c in range(channels_per_edge):
                    self.weight[i * channels_per_edge + c, j * channels_per_edge + c] = 1.0 / len(pair)

        self.weight = nn.Parameter(self.weight, requires_grad=False)

    def forward(self, input: torch.Tensor):
        return torch.matmul(self.weight, input)

class SkeletonUnpool(nn.Module):
    def __init__(self, pooling_list, channels_per_edge):
        super(SkeletonUnpool, self).__init__()
        self.pooling_list = pooling_list
        self.input_edge_num = len(pooling_list)
        self.output_edge_num = 0
        self.channels_per_edge = channels_per_edge
        for t in self.pooling_list:
            self.output_edge_num += len(t)

        self.description = 'SkeletonUnpool(in_edge_num={}, out_edge_num={})'.format(
            self.input_edge_num, self.output_edge_num,
        )

        self.weight = torch.zeros(self.output_edge_num * channels_per_edge, self.input_edge_num * channels_per_edge)

        for i, pair in enumerate(self.pooling_list):
            for j in pair:
                for c in range(channels_per_edge):
                    self.weight[j * channels_per_edge + c, i * channels_per_edge + c] = 1

        self.weight = nn.Parameter(self.weight)
        self.weight.requires_grad_(False)

    def forward(self, input: torch.Tensor):
        return torch.matmul(self.weight, input)


def calc_edge_mat(edges):
    edge_num = len(edges)
    # edge_mat[i][j] = distance between edge(i) and edge(j)
    edge_mat = [[100000] * edge_num for _ in range(edge_num)]

    for i in range(edge_num):
        edge_mat[i][i] = 0 # diagonal

    # initialize edge_mat with direct neighbor
    for i, a in enumerate(edges):
        for j, b in enumerate(edges):
            link = 0
            for x in range(2):
                for y in range(2):
                    if a[x] == b[y]:
                        link = 1
            if link:
                edge_mat[i][j] = 1 # distance = 1 

    # edge_mat: distance of all verteics pair 
    for k in range(edge_num):
        for i in range(edge_num):
            for j in range(edge_num):
                edge_mat[i][j] = min(edge_mat[i][j], edge_mat[i][k] + edge_mat[k][j]) # get distance of vertices 
    return edge_mat

def find_neighbor(edges, d):
    edge_mat = calc_edge_mat(edges)
    neighbor_list = []
    edge_num = len(edge_mat)
    for i in range(edge_num):
        neighbor = []
        for j in range(edge_num):
            if edge_mat[i][j] <= d:
                neighbor.append(j)
        neighbor_list.append(neighbor)

    # add neighbor for global part
    # why this ? 
    # meaning of 22 ? make a new root ? 
    global_part_neighbor = neighbor_list[0].copy()
    for i in global_part_neighbor:
        neighbor_list[i].append(edge_num)
    neighbor_list.append(global_part_neighbor)

    return neighbor_list

""" conv Encoder & Decoder """
class Encoder(nn.Module):
    def __init__(self, args, topology):
        super().__init__() # Encoder
        self.input_dim = args.window_size
        self.args = args 

        # topology : [parent, idx, offset ], ...
        self.topologies = [topology]
        if args.rotation == 'euler_angle': self.channel_base = [3]
        elif args.rotation == 'quaternion': self.channel_base = [4]
        self.channel_list = []
        self.edge_num = [len(topology) + 1]
        self.pooling_list = []
        self.layers = nn.ModuleList()
        self.convs = []
        add_offset = args.add_offset

        bias = True

        kernel_size = args.kernel_size
        padding = (kernel_size - 1) // 2

        for i in range(args.num_layers):
            self.channel_base.append(self.channel_base[-1] * 2)

        for i in range(args.num_layers):
            seq = []
            neighbor_list = find_neighbor(self.topologies[i], args.skeleton_dist)
            in_channels = self.channel_base[i] * self.edge_num[i]
            out_channels = self.channel_base[i+1] * self.edge_num[i]
           
            if i == 0: self.channel_list.append(in_channels)
            self.channel_list.append(out_channels)

            seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                    joint_num=self.edge_num[i], kernel_size=kernel_size, stride=2,
                                    padding=padding, padding_mode=args.padding_mode, bias=bias, add_offset=add_offset,
                                    in_offset_channel=3 * self.channel_base[i] // self.channel_base[0]))
            self.convs.append(seq[-1])
            last_pool = True if i == args.num_layers - 1 else False
            pool = SkeletonPool(edges=self.topologies[i], pooling_mode=args.skeleton_pool,
                                channels_per_edge=out_channels // len(neighbor_list), last_pool=last_pool)
            seq.append(pool)
            seq.append(nn.LeakyReLU(negative_slope=0.2))
            self.layers.append(nn.Sequential(*seq))
            
            # why need this ?
            self.topologies.append(pool.new_edges) # check topology added 
            self.pooling_list.append(pool.pooling_list)
            self.edge_num.append(len(self.topologies[-1]) + 1)
            if i == args.num_layers - 1:
                self.last_channel = self.edge_num[-1] * self.channel_base[i + 1]

    def forward(self, input, offset=None):
        # check needed ?
        # padding the one zero row to global position, so each joint including global position has 4 channels as input
        if self.args.rotation == 'quaternion' and self.args.pos_repr != '4d':
            input = torch.cat((input, torch.zeros_like(input[:, [0], :])), dim=1)

        for i, layer in enumerate(self.layers):
            # if self.args.skeleton_info == 'concat' and offset is not None:
            #     self.convs[i].set_offset(offset[i])
            input = layer(input)
        return input


class Decoder(nn.Module):
    def __init__(self, args, enc: Encoder):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.args = args
        self.enc = enc
        self.convs = []

        kernel_size = args.kernel_size
        padding = (kernel_size - 1) // 2

        if args.skeleton_info == 'concat': add_offset = True
        else: add_offset = False

        for i in range(args.num_layers):
            seq = []
            in_channels = enc.channel_list[args.num_layers - i]
            out_channels = in_channels // 2
            neighbor_list = find_neighbor(enc.topologies[args.num_layers - i - 1], args.skeleton_dist)

            if i != 0 and i != args.num_layers - 1:
                bias = False
            else:
                bias = True

            self.unpools.append(SkeletonUnpool(enc.pooling_list[args.num_layers - i - 1], in_channels // len(neighbor_list)))

            seq.append(nn.Upsample(scale_factor=2, mode=args.upsampling, align_corners=False))
            seq.append(self.unpools[-1])
            seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                    joint_num=enc.edge_num[args.num_layers - i - 1], kernel_size=kernel_size, stride=1,
                                    padding=padding, padding_mode=args.padding_mode, bias=bias, add_offset=add_offset,
                                    in_offset_channel=3 * enc.channel_base[args.num_layers - i - 1] // enc.channel_base[0]))
            self.convs.append(seq[-1])
            if i != args.num_layers - 1: seq.append(nn.LeakyReLU(negative_slope=0.2))

            self.layers.append(nn.Sequential(*seq))

    def forward(self, input, offset=None):
        for i, layer in enumerate(self.layers):
            # if self.args.skeleton_info == 'concat':
            #     self.convs[i].set_offset(offset[len(self.layers) - i - 1])
            input = layer(input)
            
        # throw the padded rwo for global position
        if self.args.rotation == 'quaternion' and self.args.pos_repr != '4d':
            input = input[:, :-1, :]

        return input


def build_edge_topology(topology, offset):
    # get all edges (pa, child, offset)
    edges = []
    joint_num = len(topology)
    for i in range(1, joint_num):
        edges.append((topology[i], i, offset[i]))
    return edges

class MotionGenerator(nn.Module):
    def __init__(self, args, offsets, joint_topology):
        super().__init__()
        self.input_dim = args.window_size
        self.joint_topology = joint_topology
        self.edges = build_edge_topology(joint_topology, torch.zeros((len(joint_topology), 3)))
        self.offsets = offsets
        # self.layers = nn.ModuleList([nn.Linear(self.input_dim, self.input_dim) for _ in range(args.n_layer)])

        # layers 
        self.encoder = Encoder(args, self.edges)
        self.decoder = Decoder(args, self.encoder)

    def forward(self, input_character, output_character, inputs):
        # Need to : change offset
        lat = self.encoder(inputs, self.offsets)
        outputs = self.decoder(lat, self.offsets)
        return outputs, lat

class Discriminator(nn.Module):
    def __init__(self, args, offsets):
        super(Discriminator, self).__init__()
        self.input_dim = args.window_size
        self.layers = nn.ModuleList([nn.Linear(self.input_dim, self.input_dim) for _ in range(args.n_layer)])

    def forward(self, input_character, output_character, inputs):
        
        for layer in self.layers:
            outputs = layer(inputs)
            
        output = outputs.reshape(outputs.shape[0], -1)

        return torch.sigmoid(output)
