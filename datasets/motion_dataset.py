from torch.utils.data import Dataset
import os
import sys
import numpy as np
import torch
from wandb import set_trace
sys.path.append("../")
from option_parser import get_std_bvh
sys.path.append("./utils")
from Quaternions import Quaternions

# for each characters 
class MotionData(Dataset):
    """
    Clip long dataset into fixed length window for batched training
    each data is a 2d tensor with shape (Joint_num*3) * Time
    """
    def __init__(self, args, positional_encoding):
        super(MotionData, self).__init__()
        self.root_pos_disp = args.root_pos_disp
        name = args.dataset # character_name
        if args.is_train == 1:
            file_path = './datasets/Mixamo/{}.npy'.format(name)
        else:
            file_path = './datasets/Mixamo/test_{}.npy'.format(name)

        print('load from file {}'.format(file_path))
        self.total_frame = 0
        self.std_bvh = get_std_bvh(args)
        self.args = args
        self.data = []
        self.motion_length = []
        motions = np.load(file_path, allow_pickle=True)
        motions = list(motions)
        
        # motions (111, 313, 69) -> new_window (192, 64,91)
        new_window = self.get_windows(motions)

        # change data from list to tensor
        self.data.append(new_window)
        self.data = torch.cat(self.data)
        
        """ change data dimensiton : (bs, window, DoF) -> (bs, DoF, winodw) """
        self.data = self.data.permute(0, 2, 1)

        """ Modify data  """
        # data: (bs, DoF, window)
        num_bs = self.data.size(0)
        num_DoF = self.data.size(1)
        num_frames = self.data.size(2)

        #  root position -> displacement 
        if self.root_pos_disp == 1:
            for bs in range(num_bs): # 0차원(motions)에 대해
                for frame in range(num_frames - 1): # 2차원(frames)에 대해. frame: 0 ~ 126
                    self.data[bs][num_DoF - 3][frame] = self.data[bs][num_DoF - 3][frame + 1] - self.data[bs][num_DoF - 3][frame]
                    self.data[bs][num_DoF - 2][frame] = self.data[bs][num_DoF - 2][frame + 1] - self.data[bs][num_DoF - 2][frame]
                    self.data[bs][num_DoF - 1][frame] = self.data[bs][num_DoF - 1][frame + 1] - self.data[bs][num_DoF - 1][frame]
                # 마지막 프레임의 disp는 0으로 셋팅해줍니다. 
                self.data[bs][num_DoF - 3][num_frames - 1] = 0
                self.data[bs][num_DoF - 2][num_frames - 1] = 0
                self.data[bs][num_DoF - 1][num_frames - 1] = 0


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
            self.mean.zero_() # 왜 이렇게 하는거지?
            self.var = torch.ones_like(self.mean)

        # Normalzation to data 
        # if args.normalization == 1:
        #     self.data = (self.data - self.mean) / self.var


        """ save data """
        train_len = self.data.shape[0] * 94 // 100
        
        # (104,91,913)
        self.data = self.data[:train_len, ...]
        self.data_reverse = torch.tensor(self.data.numpy()[..., ::-1].copy())

        # (8,91,913)
        self.test_set = self.data[train_len:, ...]       
        self.reset_length_flag = 0
        self.virtual_length = 0

        print('Data count: {}, total frame (without downsampling): {}, Noramlization: {}'.format(len(self), self.total_frame, args.normalization))

    def reset_length(self, length):
        self.reset_length_flag = 1
        self.virtual_length = length

    def __len__(self):
        if self.reset_length_flag:
            return self.virtual_length
        else:
            return self.data.shape[0]

    def __getitem__(self, item):
        if isinstance(item, int): item %= self.data.shape[0]
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

                rotations = new[:, :-1, :] # rotations: (221, 22, 3) : euler to Quaternion 
                rotations = Quaternions.from_euler(np.radians(rotations)).qs # rotations: (221, 22, 4)
                rotations = rotations.reshape(rotations.shape[0], -1) # rotations: (221, 88)

                # rotations (221,88) + positions(new[:,-1,:]) (221, 23, 3) -> (221, 91)
                new = np.concatenate((rotations, new[:, -1, :].reshape(new.shape[0], -1)), axis=1)

            # (1, frames, 91)
            new = new[np.newaxis, ...]
            new_window = torch.tensor(new, dtype=torch.float32)
            
            # add padding
            if len(motion) < max_frame:
                zero_tensor = torch.zeros((1, max_frame - new_window.size(1), new_window.size(2)))
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
            # motion = self.subsample(motion)
            self.motion_length.append(motion.shape[0])
            n_window = motion.shape[0] // step_size - 1 # 마지막 window에 데이터가 전부 차지 않았다면 제거 

            for i in range(n_window):
                begin = i * step_size
                end = begin + window_size

                # new: (64, 69)
                new = motion[begin:end, :]
                if self.args.rotation == 'quaternion':
                    new = new.reshape(new.shape[0], -1, 3)
                    rotations = new[:, :-1, :]
                    rotations = Quaternions.from_euler(np.radians(rotations)).qs
                    rotations = rotations.reshape(rotations.shape[0], -1)
                    new = np.concatenate((rotations, new[:, -1, :].reshape(new.shape[0], -1)), axis=1)
                
                new = new[np.newaxis, ...]

                # (1,64,91)
                new_window = torch.tensor(new, dtype=torch.float32)
                new_windows.append(new_window)

        return torch.cat(new_windows)

    def subsample(self, motion):
        return motion[::2, :]

    def denormalize(self, motion):
        if self.args.normalization:
            if self.var.device != motion.device:
                self.var = self.var.to(motion.device)
                self.mean = self.mean.to(motion.device)
            ans = motion * self.var + self.mean
        else: ans = motion
        return ans