from torch.utils.data import Dataset
import copy
from datasets.motion_dataset import MotionData
import os
import numpy as np
import torch
from datasets.bvh_parser import BVH_file
from option_parser import get_std_bvh
from datasets import get_test_set
from datasets import get_validation_set


class MixedData0(Dataset):
    """ Mixed data for many skeletons but one topologies """
    def __init__(self, args, motions, skeleton_idx):
        super(MixedData0, self).__init__()
        self.motions = motions
        self.motions_reverse = torch.tensor(self.motions.numpy()[..., ::-1].copy())
        self.skeleton_idx = skeleton_idx
        self.length = motions.shape[0]
        self.args = args

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if self.args.data_augment == 0 or torch.rand(1) < 0.5:
            return self.motions[item]
            # return [self.motions[item], self.skeleton_idx[item]]
        else:
            print("reversed MixedData0")
            return self.motions_reverse[item]
            # return [self.motions_reverse[item], self.skeleton_idx[item]]


class MixedData(Dataset):
    """ data_gruop_num * 2 * samples """
    def __init__(self, args, datasets_groups):
        device = torch.device(args.cuda_device if (torch.cuda.is_available()) else 'cpu')
        self.final_data = []
        self.encoded_final_data = []
        self.length = 0
        self.offsets = []
        self.joint_topologies = []
        self.ee_ids = []
        self.means = []
        self.vars = []
        dataset_num = 0
        seed = 19260817
        total_length = 10000000
        all_datas = []
        encoded_all_datas = []
        self.position_encoding = args.position_encoding

        self.split_index = []
        # all_datas : (2 groups, 4 characters, 106 motions, 913 frames, rot and pos of 91 joints)
        for group, datasets in enumerate(datasets_groups):
            offsets_group = []
            means_group = []
            vars_group = []
            dataset_num += len(datasets)
            motion_data = []
            encoded_motion_data = []
            
            # for each character in a group
            for i, dataset in enumerate(datasets):
                new_args = copy.copy(args)
                new_args.data_augment = 0
                new_args.dataset = dataset

                motion_data.append(MotionData(new_args, 0))
                encoded_motion_data.append(MotionData(new_args, self.position_encoding)) # positional encoding paramerization
                # tmp : 4 character, 106 motions, 913 frames, 111 rot + pos

                mean = np.load('./datasets/Mixamo/mean_var/{}_mean.npy'.format(dataset))
                var = np.load('./datasets/Mixamo/mean_var/{}_var.npy'.format(dataset))
                mean = torch.tensor(mean)
                mean = mean.reshape((1,) + mean.shape)
                var = torch.tensor(var)
                var = var.reshape((1,) + var.shape)

                means_group.append(mean)
                vars_group.append(var)

                file = BVH_file(get_std_bvh(dataset=dataset))
                if i == 0:
                    self.joint_topologies.append(file.topology)
                    self.ee_ids.append(file.get_ee_id())
                new_offset = file.offset
                new_offset = torch.tensor(new_offset, dtype=torch.float)
                new_offset = new_offset.reshape((1,) + new_offset.shape)
                offsets_group.append(new_offset)

                total_length = min(total_length, len(motion_data[-1]))
                
            all_datas.append(motion_data)
            encoded_all_datas.append(encoded_motion_data)

            offsets_group = torch.cat(offsets_group, dim=0)
            offsets_group = offsets_group.to(device)
            # (4,1,91,1 -> 4,91,1)
            means_group = torch.cat(means_group, dim=0).to(device)
            vars_group = torch.cat(vars_group, dim=0).to(device)

            self.offsets.append(offsets_group)
            self.means.append(means_group)
            self.vars.append(vars_group)
        
        """ Get gt_data """
        # final_data: (2, 424, 913, 91) for 2 groups
        for datasets in all_datas:
            pt = 0
            motions = []
            skeleton_idx = []

            # for each character in a group
            for dataset in datasets:
                motions.append(dataset[:])
                skeleton_idx += [pt] * len(dataset)
                pt += 1

            # (4,106, 91, 913) -> (424, 91, 913)
            motions = torch.cat(motions, dim=0)
            # length : num of total motions for 4 characters 
            if self.length != 0 and self.length != len(skeleton_idx):
                self.length = min(self.length, len(skeleton_idx))
            else:
                self.length = len(skeleton_idx)

            self.final_data.append(MixedData0(args, motions, skeleton_idx))

        """ Get positional encoded input_data """
        for datasets in encoded_all_datas:
            pt = 0
            motions = []
            skeleton_idx = []

            # for each character in a group
            for dataset in datasets:
                motions.append(dataset[:])
                skeleton_idx += [pt] * len(dataset)
                pt += 1

            # (4,106, 91, 913) -> (424, 91, 913)
            motions = torch.cat(motions, dim=0)
            # length : num of total motions for 4 characters 
            if self.length != 0 and self.length != len(skeleton_idx):
                self.length = min(self.length, len(skeleton_idx))
            else:
                self.length = len(skeleton_idx)

            self.encoded_final_data.append(MixedData0(args, motions, skeleton_idx))

    def denorm(self, gid, pid, data):
        means = self.means[gid][pid, ...]
        var = self.vars[gid][pid, ...]
        return data * var + means

    def GetDoF(self):
        DoF = len(self.final_data[0][0])
        print(f"Dof : {DoF}")
        return DoF

    # def GetMaxFrame(self):
    #     return len(self.final_data[0][0])

    def __len__(self):
        # total motion length for every character (4 * 106 = 424)
        return self.length

    def __getitem__(self, item):
        # (input motion, output motion)
        return (torch.as_tensor(self.encoded_final_data[0][item].data), # position encoded 
                torch.as_tensor(self.final_data[0][item].data)) # gt
                  


class TestData(Dataset):
    def __init__(self, args, characters):
        self.characters = characters
        self.file_list = get_test_set() 
        self.mean = []
        self.joint_topologies = []
        self.var = []
        self.offsets = []
        self.ee_ids = []
        self.args = args
        self.device = torch.device(args.cuda_device)

        for i, character_group in enumerate(characters):
            mean_group = []
            var_group = []
            offsets_group = []
            for j, character in enumerate(character_group):
                file = BVH_file(get_std_bvh(dataset=character))                
                if j == 0:
                    self.joint_topologies.append(file.topology)
                    # print(self.joint_topologies)
                    self.ee_ids.append(file.get_ee_id())
                new_offset = file.offset
                new_offset = torch.tensor(new_offset, dtype=torch.float)
                new_offset = new_offset.reshape((1,) + new_offset.shape)
                offsets_group.append(new_offset)
                mean = np.load('./datasets/Mixamo/mean_var/{}_mean.npy'.format(character))
                var = np.load('./datasets/Mixamo/mean_var/{}_var.npy'.format(character))
                mean = torch.tensor(mean)
                mean = mean.reshape((1, ) + mean.shape)
                var = torch.tensor(var)
                var = var.reshape((1, ) + var.shape)
                mean_group.append(mean)
                var_group.append(var)

            mean_group = torch.cat(mean_group, dim=0).to(self.device)
            var_group = torch.cat(var_group, dim=0).to(self.device)
            offsets_group = torch.cat(offsets_group, dim=0).to(self.device)
            self.mean.append(mean_group)
            self.var.append(var_group)
            self.offsets.append(offsets_group)
        

    def __getitem__(self, item):
        res = []
        bad_flag = 0
        for i, character_group in enumerate(self.characters):            
            res_group = []
            ref_shape = None
            for j in range(len(character_group)):
                new_motion =  self.get_item(i, j, item) # self.get_validation_item(i, j, item)
                if new_motion is not None:                    
                    new_motion = new_motion.reshape((1, ) + new_motion.shape)                                        
                    new_motion = (new_motion - self.mean[i][j]) / self.var[i][j]
                    ref_shape = new_motion                
                res_group.append(new_motion)                


            if ref_shape is None:
                print('Bad at {}'.format(item))
                return None
            for j in range(len(character_group)):
                if res_group[j] is None:
                    bad_flag = 1
                    res_group[j] = torch.zeros_like(ref_shape)
            if bad_flag:
                print('Bad at {}'.format(item))
                            
            #print("size: ({}, {}, {}, {})".format(len(res_group), len(res_group[0]), len(res_group[0][0]), len(res_group[0][0][0])))
            res_group = torch.cat(res_group, dim=0)            
            res.append([res_group, list(range(len(character_group)))])
                
        return res

    def __len__(self):
        return len(self.file_list)

    def get_item(self, gid, pid, id):
        character = self.characters[gid][pid]
        path = './datasets/Mixamo/{}/'.format(character)           
        #print("path:", path)     
        #print("id {}: {}".format(id, self.file_list[id]))
        if isinstance(id, int):
            file = path + self.file_list[id]
        elif isinstance(id, str):
            file = id
        else:
            #print(file)
            raise Exception('Wrong input file type')
        if not os.path.exists(file):
            raise Exception('Cannot find file')
        file = BVH_file(file)
        motion = file.to_tensor(quater=self.args.rotation == 'quaternion')
        motion = motion[:, ::2]
        length = motion.shape[-1]
        length = length // 4 * 4
        return motion[..., :length].to(self.device)

    def denorm(self, gid, pid, data):
        means = self.mean[gid][pid, ...]
        var = self.var[gid][pid, ...]
        return data * var + means

    def normalize(self, gid, pid, data):
        means = self.mean[gid][pid, ...]
        var = self.var[gid][pid, ...]
        return (data - means) / var

class ValidationData(Dataset):
    def __init__(self, args, characters):
        self.characters = characters
        self.file_list = get_validation_set()
        self.mean = []
        self.joint_topologies = []
        self.var = []
        self.offsets = []
        self.ee_ids = []
        self.args = args
        self.device = torch.device(args.cuda_device)

        print("loading validation data")

        for i, character_group in enumerate(characters):
            mean_group = []
            var_group = []
            offsets_group = []
            for j, character in enumerate(character_group):
                file = BVH_file(get_std_bvh(dataset=character))                
                if j == 0:
                    self.joint_topologies.append(file.topology)
                    # print(self.joint_topologies)
                    self.ee_ids.append(file.get_ee_id())
                new_offset = file.offset
                new_offset = torch.tensor(new_offset, dtype=torch.float)
                new_offset = new_offset.reshape((1,) + new_offset.shape)
                offsets_group.append(new_offset)
                mean = np.load('./datasets/Mixamo/mean_var/{}_mean.npy'.format(character)) # validation/
                var = np.load('./datasets/Mixamo/mean_var/{}_var.npy'.format(character)) # validation/
                mean = torch.tensor(mean)
                mean = mean.reshape((1, ) + mean.shape)
                var = torch.tensor(var)
                var = var.reshape((1, ) + var.shape)
                mean_group.append(mean)
                var_group.append(var)

            mean_group = torch.cat(mean_group, dim=0).to(self.device)
            var_group = torch.cat(var_group, dim=0).to(self.device)
            offsets_group = torch.cat(offsets_group, dim=0).to(self.device)
            self.mean.append(mean_group)
            self.var.append(var_group)
            self.offsets.append(offsets_group)
        

    def __getitem__(self, item):
        res = []
        bad_flag = 0
        for i, character_group in enumerate(self.characters):            
            res_group = []
            ref_shape = None
            for j in range(len(character_group)):
                new_motion = self.get_validation_item(i, j, item) # self.get_item(i, j, item)
                if new_motion is not None:                    
                    new_motion = new_motion.reshape((1, ) + new_motion.shape)                                        
                    new_motion = (new_motion - self.mean[i][j]) / self.var[i][j]
                    ref_shape = new_motion                
                res_group.append(new_motion)                


            if ref_shape is None:
                print('Bad at {}'.format(item))
                return None
            for j in range(len(character_group)):
                if res_group[j] is None:
                    bad_flag = 1
                    res_group[j] = torch.zeros_like(ref_shape)
            if bad_flag:
                print('Bad at {}'.format(item))
                            
            #print("size: ({}, {}, {}, {})".format(len(res_group), len(res_group[0]), len(res_group[0][0]), len(res_group[0][0][0])))
            res_group = torch.cat(res_group, dim=0)            
            res.append([res_group, list(range(len(character_group)))])
                
        return res

    def __len__(self):
        return len(self.file_list)

    def get_validation_item(self, gid, pid, id):
        character = self.characters[gid][pid]
        path = './datasets/Mixamo/0_Validation/{}/'.format(character)           
        # print("path: {} {}: {}".format(path, id, self.file_list[id]))

        if isinstance(id, int):
            file = path + self.file_list[id]
        elif isinstance(id, str):
            file = id
        else:
            #print(file)
            raise Exception('Wrong input file type')
        if not os.path.exists(file):
            raise Exception('Cannot find file')
        #print("file: ",file)
        file = BVH_file(file)
        motion = file.to_tensor(quater=self.args.rotation == 'quaternion')       #motion : (#frames (dim0), #joints (dim1))         

        motion = motion[:, ::2] #2개씩 나눔 
        length = motion.shape[-1] # #ofFrames
        length = length // 4 * 4 # 4의 배수로 바꿈. 
        return motion[..., :length].to(self.device)
    
    def denorm(self, gid, pid, data):
        means = self.mean[gid][pid, ...]
        var = self.var[gid][pid, ...]
        return data * var + means

    def normalize(self, gid, pid, data):
        means = self.mean[gid][pid, ...]
        var = self.var[gid][pid, ...]
        return (data - means) / var
