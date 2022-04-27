from Quaternions import Quaternions
from json.encoder import py_encode_basestring
from tracemalloc import start
from torch.utils.data import Dataset
import copy

# from wandb import set_trace
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


class MixedData0(Dataset):
    """ Mixed data for many skeletons but one topologies """

    def __init__(self, args, motions, skeleton_idx):
        super(MixedData0, self).__init__()
        self.motions = motions
        self.motions_reverse = torch.tensor(
            self.motions.numpy()[..., ::-1].copy())
        self.skeleton_idx = skeleton_idx
        self.length = motions.shape[0]
        self.args = args

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if self.args.data_augment == 0:  # or torch.rand(1) < 0.5:
            return [self.motions[item], self.skeleton_idx[item]]
        else:
            return [self.motions_reverse[item], self.skeleton_idx[item]]


class MixedData(Dataset):
    """ data_gruop_num * 2 * samples """

    def __init__(self, args, character_groups):
        self.args = args
        device = torch.device(args.cuda_device if (
            torch.cuda.is_available()) else 'cpu')
        self.final_data = []
        self.groups_body_parts_index = []
        self.body_parts_index = []

        self.skeleton_type = []
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
        self.start_pos = []
        dataset_num = 0
        total_length = 10000000
        all_datas = []
        self.offsets_group = []
        # all_datas : (2 groups, 4 characters, 106 motions, 913 frames, rot and pos of 91 joints)
        for group, characters in enumerate(character_groups):  # names
            offsets_group = []
            means_group = []
            vars_group = []
            start_pos_group = []
            dataset_num += len(characters)
            motion_data = []

            """ motion data """
            for i, character in enumerate(characters):
                args.dataset = character
                motion = MotionData(args, 0)
                motion_data.append(motion)
                total_length = min(total_length, len(motion_data[-1]))

                # 4 character, 106 motions, 913 frames, 91/111 rot + pos
                mean = np.load(
                    './datasets/Mixamo/mean_var/{}_mean.npy'.format(character))
                var = np.load(
                    './datasets/Mixamo/mean_var/{}_var.npy'.format(character))

                mean = torch.tensor(mean)
                mean = mean.reshape((1,) + mean.shape)
                var = torch.tensor(var)
                var = var.reshape((1,) + var.shape)
                start_pos = torch.tensor(motion.start_pos)
                start_pos = start_pos.reshape((1,) + start_pos.shape)

                means_group.append(mean)
                vars_group.append(var)
                start_pos_group.append(start_pos)

                file = BVH_file(get_std_bvh(dataset=character))
                if i == 0:
                    self.joint_topologies.append(file.topology)
                    self.ee_ids.append(file.get_ee_id())

                new_offset = file.offset
                new_offset = torch.tensor(new_offset, dtype=torch.float)
                new_offset = new_offset.reshape((1,) + new_offset.shape)
                offsets_group.append(new_offset)
            # (2 groups, 4 characters, 1, 91, 128)
            all_datas.append(motion_data)

            offsets_group = torch.cat(offsets_group, dim=0)
            offsets_group = offsets_group.to(device)
            self.offsets_group.append(offsets_group)
            self.offsets.append(offsets_group)

            # (4,1,91,1 -> 4,91,1)
            means_group = torch.cat(means_group, dim=0).to(device)
            vars_group = torch.cat(vars_group, dim=0).to(device)
            start_pos_group = torch.cat(start_pos_group, dim=0).to(device)
            self.means.append(means_group)
            self.vars.append(vars_group)
            self.start_pos.append(start_pos_group)

        """ Process motion: Get final_data """
        # Cropping: num_motions = batch_size * n
        num_motions = int(
            len(all_datas[0][0]) / args.batch_size) * args.batch_size
        print("num_motions for 1 epoch: ", num_motions)
        args.num_motions = num_motions

        for group_idx, datasets in enumerate(all_datas):
            pt = 0
            motions = []
            skeleton_idx = []
            for character_idx, dataset in enumerate(datasets):
                motions.append(dataset[:num_motions])
                skeleton_idx += [pt] * len(dataset)
                pt += 1
            # (4,106, 91, 913) -> (424, 91, 913)
            motions = torch.cat(motions, dim=0)
            self.length = motions.size(0)
            self.final_data.append(MixedData0(args, motions, skeleton_idx))

    def denorm(self, gid, pid, data):
        means = self.means[gid][pid, ...]
        var = self.vars[gid][pid, ...]
        data = data * var + means

        return data

    def get_offsets(self):
        return self.offsets

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        res = []
        for data in self.final_data:
            res.append(data[item])
        return res


class TestData(Dataset):
    def __init__(self, args, character_groups):
        self.args = args
        device = torch.device(args.cuda_device if (
            torch.cuda.is_available()) else 'cpu')
        self.final_data = []
        self.groups_body_parts_index = []
        self.body_parts_index = []

        self.skeleton_type = []
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
        self.start_pos = []
        dataset_num = 0
        total_length = 10000000
        all_datas = []
        self.offsets_group = []
        # all_datas : (2 groups, 4 characters, 106 motions, 913 frames, rot and pos of 91 joints)
        for group, characters in enumerate(character_groups):  # names
            offsets_group = []
            means_group = []
            vars_group = []
            start_pos_group = []
            dataset_num += len(characters)
            motion_data = []

            """ motion data """
            for i, character in enumerate(characters):
                args.dataset = character
                motion = MotionData(args, 0)
                motion_data.append(motion)
                total_length = min(total_length, len(motion_data[-1]))

                # 4 character, 106 motions, 913 frames, 91/111 rot + pos
                mean = np.load(
                    './datasets/Mixamo/mean_var/{}_mean.npy'.format(character))
                var = np.load(
                    './datasets/Mixamo/mean_var/{}_var.npy'.format(character))

                mean = torch.tensor(mean)
                mean = mean.reshape((1,) + mean.shape)
                var = torch.tensor(var)
                var = var.reshape((1,) + var.shape)
                start_pos = torch.tensor(motion.start_pos)
                start_pos = start_pos.reshape((1,) + start_pos.shape)

                means_group.append(mean)
                vars_group.append(var)
                start_pos_group.append(start_pos)

                file = BVH_file(get_std_bvh(dataset=character))
                if i == 0:
                    self.joint_topologies.append(file.topology)
                    self.ee_ids.append(file.get_ee_id())

                new_offset = file.offset
                new_offset = torch.tensor(new_offset, dtype=torch.float)
                new_offset = new_offset.reshape((1,) + new_offset.shape)
                offsets_group.append(new_offset)
            # (2 groups, 4 characters, 1, 91, 128)
            all_datas.append(motion_data)

            offsets_group = torch.cat(offsets_group, dim=0)
            offsets_group = offsets_group.to(device)
            self.offsets_group.append(offsets_group)
            self.offsets.append(offsets_group)

            # (4,1,91,1 -> 4,91,1)
            means_group = torch.cat(means_group, dim=0).to(device)
            vars_group = torch.cat(vars_group, dim=0).to(device)
            start_pos_group = torch.cat(start_pos_group, dim=0).to(device)
            self.means.append(means_group)
            self.vars.append(vars_group)
            self.start_pos.append(start_pos_group)

        """ Process motion: Get final_data """
        # Cropping: num_motions = batch_size * n
        num_motions = int(
            len(all_datas[0][0]) / args.batch_size) * args.batch_size
        print("num_motions for 1 epoch: ", num_motions)
        args.num_motions = num_motions

        for group_idx, datasets in enumerate(all_datas):
            pt = 0
            motions = []
            skeleton_idx = []
            for character_idx, dataset in enumerate(datasets):
                motions.append(dataset[:num_motions])
                skeleton_idx += [pt] * len(dataset)
                pt += 1
            # (4,106, 91, 913) -> (424, 91, 913)
            motions = torch.cat(motions, dim=0)
            self.length = motions.size(0)
            self.final_data.append(MixedData0(args, motions, skeleton_idx))

    def denorm(self, gid, pid, data):
        means = self.means[gid][pid, ...]
        var = self.vars[gid][pid, ...]
        data = data * var + means

        return data

    def get_offsets(self):
        return self.offsets

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        res = []
        for data in self.final_data:
            res.append(data[item])
        return res
