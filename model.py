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

""" Motion Generator """
class MotionGenerator(nn.Module):
    def __init__(self, args, offsets, joint_topology):
        super().__init__()
        self.edges = build_edge_topology(joint_topology, torch.zeros((len(joint_topology), 3)))
        self.fk = ForwardKinematics(args, self.edges)
        self.body_part_generator = []
        self.body_part_discriminator = []
        for i in range(6):
            body_part_generator = BodyPartGenerator(args, self.edges).to(args.cuda_device)
            self.body_part_generator.append(body_part_generator)
        for i in range(6):
            body_part_discriminator = BodyPartDiscriminator(args, self.edges).to(args.cuda_device)
            self.body_part_discriminator.append(body_part_discriminator)

    def forward(self, i, input_character, output_character, inputs):
        outputs, lat = self.body_part_generator[i](input_character, output_character, inputs)
        return outputs, lat

    def G_parameters(self):
        return list(self.body_part_generator[0].parameters())\
            +list(self.body_part_generator[1].parameters())\
                +list(self.body_part_generator[2].parameters())\
                    +list(self.body_part_generator[3].parameters())\
                        +list(self.body_part_generator[4].parameters())\
                            +list(self.body_part_generator[5].parameters())

    def D_parameters(self):
        return list(self.body_part_discriminator[0].parameters())\
            +list(self.body_part_discriminator[1].parameters())\
                +list(self.body_part_discriminator[2].parameters())\
                    +list(self.body_part_discriminator[3].parameters())\
                        +list(self.body_part_discriminator[4].parameters())\
                            +list(self.body_part_discriminator[5].parameters())

def build_edge_topology(topology, offset):
    # get all edges (pa, child, offset)
    edges = []
    joint_num = len(topology)
    for i in range(1, joint_num):
        edges.append((topology[i], i, offset[i]))
    return edges

class BodyPartGenerator(nn.Module):
    def __init__(self, args, edges):
        super().__init__()
        self.input_dim = args.window_size
        self.edges = edges

        # layers 
        self.encoder = Encoder(args, self.edges)
        self.decoder = Decoder(args, self.encoder)

    def forward(self, input_character, output_character, inputs):
        # Need to : change offset
        # remove character index 
        lat = self.encoder(inputs)
        outputs = self.decoder(lat)
        return outputs, lat

""" Conv-based Encoder & Decoder """
class Encoder(nn.Module):
    def __init__(self, args, topology):
        super().__init__() # Encoder
        self.input_dim = args.window_size
        self.args = args 

        # topology : [parent, idx, offset ], ...
        self.topologies = [topology]
        if args.rotation == 'euler_angle': self.channel_base = [3]
        elif args.rotation == 'quaternion': 
            if args.swap_dim == 1:
                self.channel_base = [4]
            else:
                self.channel_base = [args.window_size] # [4]
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
            if args.swap_dim == 1:
                in_channels = self.channel_base[i] * self.edge_num[i]
                out_channels = self.channel_base[i+1] * self.edge_num[i]
            else: 
                in_channels = self.channel_base[i] 
                out_channels = self.channel_base[i+1]

            if i == 0: self.channel_list.append(in_channels)
            self.channel_list.append(out_channels)

            seq.append(SkeletonConv(self.args, neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                    joint_num=self.edge_num[i], kernel_size=kernel_size, stride=2,
                                    padding=padding, padding_mode=args.padding_mode, bias=bias, add_offset=add_offset,
                                    in_offset_channel=3 * self.channel_base[i] // self.channel_base[0]))
            self.convs.append(seq[-1])
            last_pool = True if i == args.num_layers - 1 else False
            pool = SkeletonPool(args, edges=self.topologies[i], pooling_mode=args.skeleton_pool,
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
        # padding the one zero row to global position, so each joint including global position has 4 channels as input
        if self.args.rotation == 'quaternion' and self.args.pos_repr != '4d':
            if self.args.swap_dim == 1:
                input = torch.cat((input, torch.zeros_like(input[:, [0], :])), dim=1)
            else : 
                input = torch.cat((input, torch.zeros_like(input[:, :, [0]])), dim=2)

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
            seq.append(SkeletonConv(self.args, neighbor_list, in_channels=in_channels, out_channels=out_channels,
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

""" Conv / Pooling """
class SkeletonConv(nn.Module):
    def __init__(self, args, neighbour_list, in_channels, out_channels, kernel_size, joint_num, stride=1, padding=0,
                 bias=True, padding_mode='zeros', add_offset=False, in_offset_channel=0):
                 # padding_mode = reflection
        self.args = args 
        self.in_channels_per_joint = in_channels // joint_num
        self.out_channels_per_joint = out_channels // joint_num
        if args.swap_dim == 1:
            if in_channels % joint_num != 0 or out_channels % joint_num != 0:
                raise Exception('BAD')
        else:
            if in_channels % args.window_size != 0 or out_channels % args.window_size != 0:
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
            tmp = torch.zeros_like(self.weight[self.out_channels_per_joint * i:self.out_channels_per_joint * (i + 1),
                                   neighbour, ...]) 
            nn.init.kaiming_uniform_(tmp, a=math.sqrt(5))
            self.weight[self.out_channels_per_joint * i:self.out_channels_per_joint * (i + 1),
                        neighbour, ...] = tmp
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    self.weight[self.out_channels_per_joint * i:self.out_channels_per_joint * (i + 1), neighbour, ...])
                bound = 1 / math.sqrt(fan_in)
                tmp = torch.zeros_like(
                    self.bias[self.out_channels_per_joint * i:self.out_channels_per_joint * (i + 1)])
                nn.init.uniform_(tmp, -bound, bound)
                self.bias[self.out_channels_per_joint * i:self.out_channels_per_joint * (i + 1)] = tmp

        self.weight = nn.Parameter(self.weight)
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias)

    def set_offset(self, offset):
        if not self.add_offset: raise Exception('Wrong Combination of Parameters')
        self.offset = offset.reshape(offset.shape[0], -1)
 
    def forward(self, input):
        weight_masked = self.weight * self.mask

        res = F.conv1d(F.pad(input, self._padding_repeated_twice, mode='reflect'), # 
                       weight_masked, self.bias, self.stride,
                       0, self.dilation, self.groups)
        return res

class SkeletonPool(nn.Module):
    def __init__(self, args, edges, pooling_mode, channels_per_edge, last_pool=False):
        super(SkeletonPool, self).__init__()
        self.args = args 

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
        if self.args.swap_dim == 1:
            self.weight = torch.zeros(len(self.pooling_list) * channels_per_edge, self.edge_num * channels_per_edge)
        else: 
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

class BodyPartDiscriminator(nn.Module):
    def __init__(self, args, topology):        
        super(BodyPartDiscriminator, self).__init__()
        self.topologies = [topology]
        self.channel_base = [4] # TODO : 3 position
        self.channel_list = []
        self.joint_num = [len(topology) + 1]
        self.pooling_list = []
        self.layers = nn.ModuleList()
        self.args = args

        kernel_size = args.kernel_size
        padding = (kernel_size - 1) // 2

        for i in range(args.num_layers):
            self.channel_base.append(self.channel_base[-1] * 2)
        
        # 동일한 encoder을 사용하는 방법으로 바꾸기.
        for i in range(args.num_layers):
            seq = []
            neighbor_list = find_neighbor(self.topologies[i], args.skeleton_dist)
            in_channels = self.channel_base[i] * self.joint_num[i]
            out_channels = self.channel_base[i+1] * self.joint_num[i]
            if i == 0: self.channel_list.append(in_channels)
            self.channel_list.append(out_channels)

            if i < args.num_layers - 1: bias = False
            else: bias = True

            if i == args.num_layers - 1:
                kernel_size = 16
                padding = 0

            seq.append(SkeletonConv(self.args, neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                    joint_num=self.joint_num[i], kernel_size=kernel_size, stride=2, padding=padding,
                                    padding_mode='reflection', bias=bias))
            if i < args.num_layers - 1: seq.append(nn.BatchNorm1d(out_channels))
            pool = SkeletonPool(self.args, edges=self.topologies[i], pooling_mode=args.skeleton_pool,
                                channels_per_edge=out_channels // len(neighbor_list))
            seq.append(pool)
            if i < args.num_layers - 1:
                seq.append(nn.LeakyReLU(negative_slope=0.2))
            self.layers.append(nn.Sequential(*seq))

            self.topologies.append(pool.new_edges)
            self.pooling_list.append(pool.pooling_list)
            self.joint_num.append(len(pool.new_edges) + 1)
            if i == args.num_layers - 1:
                self.last_channel = self.joint_num[-1] * self.channel_base[i+1]

        # if not args.patch_gan: self.compress = nn.Linear(in_features=self.last_channel, out_features=1)

    def forward(self, input):
        # input = input.reshape(input.shape[0], input.shape[1], -1)
        # input = input.permute(0, 2, 1)
        if self.args.rotation == 'quaternion' and self.args.pos_repr != '4d':
            if self.args.swap_dim == 1:
                input = torch.cat((input, torch.zeros_like(input[:, [0], :])), dim=1)
            else : 
                input = torch.cat((input, torch.zeros_like(input[:, :, [0]])), dim=2)

        for layer in self.layers:
            input = layer(input)
        
        # if not self.args.patch_gan:
        #     input = input.reshape(input.shape[0], -1)
            # input = self.compress(input)
        # shape = (64, 72, 9)
        return torch.sigmoid(input).squeeze()

