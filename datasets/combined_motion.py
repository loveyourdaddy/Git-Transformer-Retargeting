from json.encoder import py_encode_basestring
from torch.utils.data import Dataset
import copy

from wandb import set_trace
from datasets.motion_dataset import MotionData
import os
import numpy as np
import torch
from datasets.bvh_parser import BVH_file
from option_parser import get_std_bvh
from datasets import get_test_set
from datasets import get_validation_set
import sys
sys.path.append("./utils")
from Quaternions import Quaternions

class MixedData0(Dataset):
    """ Mixed data for many skeletons but one topologies """
    def __init__(self, args, motions):
        super(MixedData0, self).__init__()
        self.motions = motions
        self.motions_reverse = torch.tensor(self.motions.numpy()[..., ::-1].copy())
        # self.skeleton_idx = skeleton_idx
        self.length = motions.shape[0]
        self.args = args

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if self.args.data_augment == 0: # or torch.rand(1) < 0.5:
            return self.motions[item]
        else:
            print("reversed MixedData0")
            return self.motions_reverse[item]


""" MixedData:  """
class MixedData(Dataset):
    """ data_gruop_num * 2 * samples """
    def __init__(self, args, character_groups): # characters
        self.args = args
        device = torch.device(args.cuda_device if (torch.cuda.is_available()) else 'cpu')
        self.final_data = []
        self.enc_inputs = []
        self.dec_inputs = []
        self.gt = []

        self.encoded_final_data = []
        self.length = 0
        self.offsets = []
        self.joint_topologies = []
        self.ee_ids = []
        self.means = []
        self.vars = []
        dataset_num = 0
        total_length = 10000000
        all_datas = []
        # self.position_encoding = args.position_encoding
        self.offsets_group = []
        # self.split_index = []

        # all_datas : (2 groups, 4 characters, 106 motions, 913 frames, rot and pos of 91 joints)
        for group, characters in enumerate(character_groups): # names 
            offsets_group = []
            means_group = []
            vars_group = []
            dataset_num += len(characters)
            motion_data = []

            for i, character in enumerate(characters):
                args.dataset = character
                motion = MotionData(args, 0)
                motion_data.append(motion)
                total_length = min(total_length, len(motion_data[-1]))

                # 4 character, 106 motions, 913 frames, 111 rot + pos
                mean = np.load('./datasets/Mixamo/mean_var/{}_mean.npy'.format(character))
                var = np.load('./datasets/Mixamo/mean_var/{}_var.npy'.format(character))
                mean = torch.tensor(mean)
                mean = mean.reshape((1,) + mean.shape)
                var = torch.tensor(var)
                var = var.reshape((1,) + var.shape)

                means_group.append(mean)
                vars_group.append(var)

                file = BVH_file(get_std_bvh(dataset=character))
                if i == 0:
                    self.joint_topologies.append(file.topology)
                    self.ee_ids.append(file.get_ee_id())

                new_offset = file.offset
                new_offset = torch.tensor(new_offset, dtype=torch.float)
                new_offset = new_offset.reshape((1,) + new_offset.shape)
                offsets_group.append(new_offset)
            all_datas.append(motion_data)

            offsets_group = torch.cat(offsets_group, dim=0)
            offsets_group = offsets_group.to(device)
            self.offsets_group.append(offsets_group)
            self.offsets.append(offsets_group)

            # (4,1,91,1 -> 4,91,1)
            means_group = torch.cat(means_group, dim=0).to(device)
            vars_group = torch.cat(vars_group, dim=0).to(device)
            self.means.append(means_group)
            self.vars.append(vars_group)
        
        """ Get final """
        for group_idx, datasets in enumerate(all_datas): # final_data: (2, 424, 913, 91) for 2 groups
            motions = []
            # 총 모션의 갯수가  batch_size의 배수가 되게 하기 위한 Cropping
            num_motions = int(len(datasets[0]) / args.batch_size) * args.batch_size
            print("max_length: ", num_motions)
            args.num_motions = num_motions
            
            for character_idx, dataset in enumerate(datasets): # for each character in a group
                motions.append(dataset[:num_motions])

            # (4,106, 91, 913) -> (424, 91, 913),  (4,51,138,128) -> (204,138,128)
            motions = torch.cat(motions, dim=0)
            self.length = motions.size(0)
            self.final_data.append(MixedData0(args, motions))
        
        """ Get enc / dec input motions """
        self.gt = self.final_data[1][:] 
        self.enc_inputs = self.final_data[0][:] 
        self.dec_inputs = self.final_data[1][:] 
        
        """ update input/output dimension of network """
        #swap_dim=0: (bs, Window, DoF)
        if args.swap_dim == 0:
            args.input_size = self.enc_inputs.size(2)
            args.output_size = self.dec_inputs.size(2)
        #swap_dim=1: (bs, DoF, Window)
        else: 
            args.input_size = self.enc_inputs.size(1)
            args.output_size = self.dec_inputs.size(1)
        
    def denorm(self, gid, pid, data):
        means = self.means[gid][pid, ...]
        var = self.vars[gid][pid, ...]
        # data_tmp = data 
        data = data * var + means

        # if self.args.root_pos_disp == 1: 
        #     if self.args.swap_dim == 0: #(bs, frame, DoF)
        #         data[:,:,-3:] = data_tmp[:,:,-3:]
        #     else:  #(bs, DoF, frame)
        #         data[:,-3:,:] = data_tmp[:,-3:,:]
        return data

    def get_offsets(self):
        return self.offsets

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return (torch.as_tensor(self.enc_inputs[item].data),    # source motion
                torch.as_tensor(self.dec_inputs[item].data),    # decoder source motion
                torch.as_tensor(self.gt[item].data)) # gt target motion


class TestData(Dataset):
    def __init__(self, args, character_groups):
        # self.characters = characters
        # self.file_list = get_test_set()
        self.args = args
        self.device = torch.device(args.cuda_device)
        self.final_data = []
        all_datas = []
        self.offsets = []
        self.means = []
        self.vars = []

        for i, characters in enumerate(character_groups):
            motion_data = []
            offsets_group = []
            means_group = []
            vars_group = []

            for j, character in enumerate(characters):
                file = BVH_file(get_std_bvh(dataset=character))
                args.dataset = character
                motion = MotionData(args, 0)
                motion_data.append(motion)
                
                new_offset = file.offset
                new_offset = torch.tensor(new_offset, dtype=torch.float)
                new_offset = new_offset.reshape((1,) + new_offset.shape)
                offsets_group.append(new_offset)

                # get mean and var 
                mean = np.load('./datasets/Mixamo/mean_var/{}_mean_test.npy'.format(character))
                var = np.load('./datasets/Mixamo/mean_var/{}_var_test.npy'.format(character))
                mean = motion.mean
                var = motion.var
                mean = torch.tensor(mean)
                mean = mean.reshape((1,) + mean.shape)
                var = torch.tensor(var)
                var = var.reshape((1,) + var.shape)

                means_group.append(mean)
                vars_group.append(var)

            all_datas.append(motion_data)
            # offsets_group = torch.cat(offsets_group, dim=0)
            # offsets_group = offsets_group.to(self.device)
            self.offsets.append(offsets_group)

            means_group = torch.cat(means_group, dim=0).to(self.device)
            vars_group = torch.cat(vars_group, dim=0).to(self.device)

            self.offsets.append(offsets_group)
            self.means.append(means_group)
            self.vars.append(vars_group)

        """ Get final """
        for group_idx, datasets in enumerate(all_datas): 
            motions = []
            max_length = int( len(datasets[0]) / args.batch_size) * args.batch_size 
            args.num_motions = max_length
            for character_idx, dataset in enumerate(datasets):
                motions.append(dataset[:max_length])

            # (4,106, 91, 913) -> (424, 91, 913),  (4,51,138,128) -> (204,138,128)
            motions = torch.cat(motions, dim=0)          
            self.length = motions.size(0)
            self.final_data.append(MixedData0(args, motions))
        
        """ Get enc / dec input motions """
        self.gt = self.final_data[1][:] 
        self.enc_inputs = self.final_data[0][:] 
        self.dec_inputs = self.final_data[1][:] 
        
        """ update input/output dimension of network """
        # swap == 0: input / output DoF
        # swap == 1: window_size 
        args.input_size = self.enc_inputs.size(2)
        args.output_size = self.dec_inputs.size(2)

    def denorm(self, gid, pid, data):
        means = self.means[gid][pid, ...]
        var = self.vars[gid][pid, ...]
        # data_tmp = data 
        data = data * var + means

        # if self.args.root_pos_disp == 1: 
        #     if self.args.swap_dim == 0: #(bs, frame, DoF)
        #         data[:,:,-3:] = data_tmp[:,:,-3:]
        #     else:  #(bs, DoF, frame)
        #         data[:,-3:,:] = data_tmp[:,-3:,:]
        return data


    def get_offsets(self):
        return self.offsets

    
    def __getitem__(self, item):
        return (torch.as_tensor(self.enc_inputs[item].data),    # source motion
                torch.as_tensor(self.dec_inputs[item].data),    # decoder source motion
                torch.as_tensor(self.gt[item].data)) # gt target motion

    def __len__(self):
        return self.length
