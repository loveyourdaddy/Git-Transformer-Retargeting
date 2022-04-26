from Quaternions import Quaternions
from option_parser import get_std_bvh
from torch.utils.data import Dataset
import os
import sys
import numpy as np
import torch
# from wandb import set_trace
sys.path.append("../")
sys.path.append("./utils")

# for each characters


class MotionData(Dataset):
    """
    Clip long dataset into fixed length window for batched training
    each data is a 2d tensor with shape (Joint_num*3) * Time
    """

    def __init__(self, args, preprocess):
        super(MotionData, self).__init__()
        name = args.dataset

        # Load all motion files
        file_path = './datasets/Mixamo/{}.npy'.format(name)
        # if args.is_train == 1:
        #     file_path = './datasets/Mixamo/{}.npy'.format(name)
        # else:
        #     file_path = './datasets/Mixamo/{}_test.npy'.format(name)

        print('load from file {}'.format(file_path))
        self.total_frame = 0
        self.std_bvh = get_std_bvh(args)
        self.args = args
        self.data = []
        self.motion_length = []
        motions = np.load(file_path, allow_pickle=True)
        motions = list(motions)

        print(file_path)
        new_window = self.get_windows(motions)

        # change data from list to tensor
        self.data.append(new_window)
        self.data = torch.cat(self.data)

        """ Swap dimension: (bs, Windows, Joint) -> (bs, joint, windows) """
        if args.swap_dim == 1:
            self.data = torch.transpose(self.data, 1, 2)

        """ Modify data  """
        num_DoF = self.data.size(1)
        num_frames = self.data.size(2)

        self.start_pos = self.data[:, -3:, 0]

        # root position -> displacement
        if args.root_pos_as_disp == 1:
            for frame in range(num_frames - 1):
                self.data[:, num_DoF-3:, frame] \
                    = self.data[:, num_DoF -3:, frame + 1] - self.data[:, num_DoF-3:, frame]
            self.data[:, num_DoF-3:, num_frames - 1] = 0

        """ normalization data:  mean, var of data & normalization """
        if args.normalization:
            self.mean = torch.mean(self.data, (0, 2), keepdim=True)
            self.var = torch.var(self.data, (0, 2), keepdim=True)
            self.var = self.var ** (1/2)
            idx = self.var < 1e-5
            self.var[idx] = 1
            self.data = (self.data - self.mean) / self.var
        else:
            self.mean = torch.mean(self.data, (0, 2), keepdim=True)
            self.mean.zero_()
            self.var = torch.ones_like(self.mean)

        self.reset_length_flag = 0
        self.virtual_length = 0

        print('Data count: {}, total frame (without downsampling): {}, Noramlization: {}'.format(
            len(self), self.total_frame, args.normalization))

    def reset_length(self, length):
        self.reset_length_flag = 1
        self.virtual_length = length

    def __len__(self):
        if self.reset_length_flag:
            return self.virtual_length
        else:
            return self.data.shape[0]

    def __getitem__(self, item):
        if isinstance(item, int):
            item %= self.data.shape[0]
        if self.args.data_augment == 0 or np.random.randint(0, 2) == 0:
            return self.data[item]
        else:
            print("reversed")
            return self.data_reverse[item]

    def get_motion_data(self, motions, max_frame):
        ret_motion = []

        for motion in motions:
            self.total_frame += motion.shape[0]
            self.motion_length.append(motion.shape[0])

            # new: (221, 69) : 22*3 + 3
            new = motion
            if self.args.rotation == 'quaternion':
                # new: (221, 23, 3)
                new = new.reshape(new.shape[0], -1, 3)

                # rotations: (221, 22, 3) : euler to Quaternion
                rotations = new[:, :-1, :]
                rotations = Quaternions.from_euler(
                    np.radians(rotations)).qs  # rotations: (221, 22, 4)
                rotations = rotations.reshape(
                    rotations.shape[0], -1)  # rotations: (221, 88)

                # rotations (221,88) + positions(new[:,-1,:]) (221, 23, 3) -> (221, 91)
                new = np.concatenate(
                    (rotations, new[:, -1, :].reshape(new.shape[0], -1)), axis=1)

            # (1, frames, 91)
            new = new[np.newaxis, ...]
            new_window = torch.tensor(new, dtype=torch.float32)

            # add padding
            if len(motion) < max_frame:
                zero_tensor = torch.zeros(
                    (1, max_frame - new_window.size(1), new_window.size(2)))
                new_window = torch.cat((new_window, zero_tensor), 1)

            ret_motion.append(new_window)

        # ret_motion : 111 motions,  1 new axis, 913 frames, 91 rot + pos
        return torch.cat(ret_motion)

    def get_max_frame(self, motions):
        num_motions = len(motions)
        max_length = 0
        for i in range(num_motions):
            if len(motions[i]) > max_length:
                max_length = len(motions[i])
        return max_length

    def get_windows(self, motions):
        new_windows = []
        step_size = self.args.window_size // 2
        window_size = step_size * 2

        # motions : (motions, frames, joint DoF)
        for motion in motions:
            self.total_frame += motion.shape[0]
            # motion = self.subsample(motion) # motion subsampling
            self.motion_length.append(motion.shape[0])
            n_window = motion.shape[0] // step_size - \
                1  # -1 : 마지막 window에 데이터가 전부 차지 않았다면 제거

            for i in range(n_window):
                begin = i * step_size
                end = begin + window_size

                # new: (64, 69)
                new = motion[begin:end, :]
                if self.args.rotation == 'quaternion':
                    new = new.reshape(new.shape[0], -1, 3)
                    rotations = new[:, :-1, :]
                    rotations = Quaternions.from_euler(
                        np.radians(rotations)).qs
                    rotations = rotations.reshape(rotations.shape[0], -1)
                    new = np.concatenate(
                        (rotations, new[:, -1, :].reshape(new.shape[0], -1)), axis=1)

                new = new[np.newaxis, ...]  # (1,64,91)
                new_window = torch.tensor(new, dtype=torch.float32)
                new_windows.append(new_window)

        return torch.cat(new_windows)

    def subsample(self, motion):
        return motion[::2, :]
