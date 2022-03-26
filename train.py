import torch
import os
import numpy as np
from datasets import get_character_names
# from model import ProjectionNet
import option_parser
from tqdm import tqdm
from datasets.bvh_parser import BVH_file
from datasets.bvh_writer import BVH_writer
from models.Kinematics import ForwardKinematics
# from rendering import *
import torchvision
from models.utils import GAN_loss
from model import MotionGenerator

SAVE_ATTENTION_DIR = "attention_vis"
SAVE_ATTENTION_DIR_INTRA = "attention_vis_intra"
os.makedirs(SAVE_ATTENTION_DIR, exist_ok=True)
os.makedirs(SAVE_ATTENTION_DIR_INTRA, exist_ok=True)

def remake_root_position_from_displacement(args, motions, num_bs, num_frame, num_DoF):
    for bs in range(num_bs):  # dim 0
        for frame in range(num_frame - 1):  # dim 2 # frame: 0~62. update 1 ~ 63
            motions[bs][frame + 1][num_DoF - 3] += motions[bs][frame][num_DoF - 3]
            motions[bs][frame + 1][num_DoF - 2] += motions[bs][frame][num_DoF - 2]
            motions[bs][frame + 1][num_DoF - 1] += motions[bs][frame][num_DoF - 1]

    return motions

def try_mkdir(path):
    if not os.path.exists(path):
        # print('make new dir')
        os.system('mkdir -p {}'.format(path))

def requires_grad_(model, requires_grad):
    for para in model.parameters():
        para.requires_grad = requires_grad

class GeneralModel():
    def __init__(self, args, character_names, dataset):
        # super(GeneralModel, self).__init__(args)
        self.args = args
        self.dataset = dataset
        self.characters = character_names
        offsets = dataset.get_offsets()
        self.n_topology = self.args.n_topology

        """ Models """ 
        self.modelGs = []
        self.modelDs = []
        self.optimizerGs = []
        self.optimizerDs = []
        for i in range(self.args.n_topology):
            modelG = MotionGenerator(args, offsets, dataset.joint_topologies[i]).to(args.cuda_device)
            self.modelGs.append(modelG)
            optimizerG = torch.optim.Adam(self.modelGs[i].G_parameters(), lr=args.learning_rate) # weight_decay=args.weight_decay
            self.optimizerGs.append(optimizerG)

        """ Set BVH writers """ 
        BVHWriters = []
        Files = []
        for i in range(len(character_names)):
            bvh_writers = []
            files = []
            for j in range(len(character_names[0])):
                file = BVH_file(option_parser.get_std_bvh(dataset=character_names[i][j]))
                files.append(file)
                bvh_writers.append(BVH_writer(file.edges, file.names))

            Files.append(files)
            BVHWriters.append(bvh_writers)
    
        """ define lists"""
        self.DoF        = [] # [0] * self.n_topology
        self.offset_idx = [0] * self.n_topology

        # pos 
        self.output_pos         = [0] * self.n_topology
        self.gt_pos             = [0] * self.n_topology
        self.output_pos_global  = [0] * self.n_topology
        self.gt_pos_global      = [0] * self.n_topology

        # latent_feature         = [0] * n_topology


        """ Criternion """
        self.rec_criterion = torch.nn.MSELoss() 
        # self.gan_criterion = GAN_loss(args.gan_mode).to(args.cuda_device)

        # index
        self.character_idx = 0 
        self.motion_idx = 0 

    def train_epoch(self, epoch, data_loader, save_name):
        self.epoch = epoch
        save_dir = self.args.save_dir + save_name
        try_mkdir(save_dir)

        self.prev_setting()

        for i in range(self.n_topology):
            self.modelGs[i].train()

        with tqdm(total=len(data_loader), desc=f"TrainEpoch {epoch}") as pbar:
            for i, value in enumerate(data_loader):

                self.iter_setting(i)
                self.forward(value)
                self.backward()

                if self.epoch % 100 == 0:
                    self.bvh_writing(save_dir)

                """ show info """
                pbar.update(1)
                pbar.set_postfix_str(f"Total: {np.mean(self.rec_losses):.3f}, Rec: {np.mean(self.rec_losses1):.3f}, RootPos: {np.mean(self.rec_losses2):.3f}, Global: {np.mean(self.rec_losses3):.3f}")

                # clear
    
    def prev_setting(self):
        """ update input/output dimension of network: Set DoF """
        for j in range(self.args.n_topology):
            motion = self.dataset[0][j][0] # (256, 4, motion/offset 2, 91 or 111, 128)
            self.DoF.append(motion.size(0))
    
        # if self.args.swap_dim == 0: # (bs, Window, DoF)
        #     # self.input_size  = motion.size(2)
        #     # self.output_size = motion.size(2)
        #     self.DoF[j] = motion.size(2)
        # else: # (bs, DoF, Windows)
        #     # self.input_size  = motion.size(1)
        #     # self.output_size = motion.size(1)
        #     self.DoF[j] = motion.size(1)

        # self.bp_motions[j] = torch.zeros(self.args.batch_size, 6, self.DoF[j], window_size).to(self.args.cuda_device) 

        # make output shape
        # self.bp_output_motions[j] = torch.zeros(self.bp_motions[j].size()).to(self.args.cuda_device)
        # self.output_motions[j]    = torch.zeros(self.args.batch_size, self.DoF[j], window_size).to(self.args.cuda_device)
        # self.gt_motions[j]        = torch.zeros(self.args.batch_size, self.DoF[j], window_size).to(self.args.cuda_device)
    
    def iter_setting(self, i):
        self.motion_idx = self.get_curr_motion(i, self.args.batch_size)
        self.character_idx = self.get_curr_character(self.motion_idx, self.args.num_motions) 

        self.output_motions         = []
        self.gt_motions             = []
        
        # bp
        self.bp_motions             = []
        self.bp_output_motions      = []

        # denorm
        self.denorm_gt_motions      = []
        self.denorm_output_motions  = []
        # self.denorm_gt_motions_     = []
        # self.denorm_output_motions_ = []

        # set loss list to mean (append)
        self.rec_loss =  []

        # 2 models loss for recording 
        self.rec_losses = [] 
        self.rec_losses1 = [] 
        self.rec_losses2 = [] 
        self.rec_losses3 = [] 

    def forward(self, value):

        self.separate_bp_motion(value)
        self.feed_to_network()
        self.combine_full_motion()
        self.denorm_motion()
        self.get_loss()

    def separate_bp_motion(self, value):
        for j in range(self.args.n_topology):
            motions, self.offset_idx[j] = value[j]
            motions = motions.to(self.args.cuda_device)
            
            bp_motions = torch.zeros(self.args.batch_size, 6, self.DoF[j], self.args.window_size).to(self.args.cuda_device) 

            """ body part motions for source motion """
            # body part: (:, 0~4, 0~90, :)
            for b, body_part_index in enumerate(self.dataset.groups_body_parts_index[0][self.character_idx]):
                bp_motions[:, b, body_part_index, :] = motions[:, body_part_index, :]
            # root rotation (:, 0, 0~91, :)
            for quat_idx in range(4):
                bp_motions[:, 5, quat_idx, :] = motions[:, quat_idx, :]
            # root position : 4~7
            for quat_idx in range(3):
                bp_motions[:, 5, self.DoF[j] - 3 + quat_idx, :] = motions[:, self.DoF[j] - 3 + quat_idx, :]

            self.bp_motions.append(bp_motions)

    def feed_to_network(self):
        """ feed to NETWORK """
        for j in range(self.args.n_topology):
            bp_output_motions = torch.zeros(self.bp_motions[j].size()).to(self.args.cuda_device)
            for b in range(6):
                bp_output_motions[:,b,:,:], _ = self.modelGs[j].body_part_generator[b](self.character_idx, self.character_idx, self.bp_motions[j][:, b, :, :])
            self.bp_output_motions.append(bp_output_motions)

    def combine_full_motion(self):
        """ Combine part by motion back """              
        index = self.dataset.groups_body_parts_index 
        for j in range(self.args.n_topology):
            
            gt_motions     = torch.zeros(self.args.batch_size, self.DoF[j], self.args.window_size).to(self.args.cuda_device)
            output_motions = torch.zeros(self.args.batch_size, self.DoF[j], self.args.window_size).to(self.args.cuda_device)

            for b in range(5): # body parts
                gt_motions[:, index[j][self.character_idx][b], :]     = self.bp_motions[j][:, b, index[j][self.character_idx][b], :]
                output_motions[:, index[j][self.character_idx][b], :] = self.bp_output_motions[j][:, b, index[j][self.character_idx][b], :]
            for k in range(4): # root rotation (0~3)
                gt_motions[:, k, :]      = self.bp_motions[j][:, 5, k, :]
                output_motions[:, k, :]  = self.bp_output_motions[j][:, 5, k, :]
            for k in range(3): # position (len-3 ~ -1)
                gt_motions[:, self.DoF[j] - 3 + k, :]     = self.bp_motions[j][:, 5, self.DoF[j] - 3 + k, :]
                output_motions[:, self.DoF[j] - 3 + k, :] = self.bp_output_motions[j][:, 5, self.DoF[j] - 3 + k, :]

            self.gt_motions.append(gt_motions)
            self.output_motions.append(output_motions)

    def denorm_motion(self):
        """ Denorm and transpose & Remake root & Get global position """
        for j in range(self.args.n_topology):
            
            if self.args.normalization == 1:
                denorm_gt_motions     = self.denormalize(self.character_idx, self.gt_motions[j], j)
                denorm_output_motions = self.denormalize(self.character_idx, self.output_motions[j], j)
            else:
                denorm_gt_motions     = self.gt_motions[j]
                denorm_output_motions = self.output_motions[j]

            self.denorm_gt_motions.append(denorm_gt_motions)
            self.denorm_output_motions.append(denorm_output_motions)

    def get_loss(self):
        """ loss1. reconstruction loss on each element """
        for j in range(self.args.n_topology):
            # loss1-1. on each element
            rec_loss1 = self.rec_criterion(self.gt_motions[j], self.output_motions[j])
            self.rec_losses1.append(rec_loss1.item())

            # loss 1-2. root position / height
            # rec_loss2 = self.rec_criterion(self.denorm_gt_motions[j][:, -3:, :], self.denorm_output_motions[j][:, -3:, :]) # / height
            # self.rec_losses2.append(rec_loss2.item())

            # # loss 1-3. global position
            # self.output_pos[j]        = self.modelGs[j].fk.forward_from_raw   (self.denorm_output_motions[j], self.dataset.offsets[j][self.offset_idx[j]])
            # self.gt_pos[j]            = self.modelGs[j].fk.forward_from_raw   (self.denorm_gt_motions[j], self.dataset.offsets[j][self.offset_idx[j]]).detach()
            # self.output_pos_global[j] = self.modelGs[j].fk.from_local_to_world(self.output_pos[j]) # / height.reshape(height.shape + (1, ))
            # self.gt_pos_global[j]     = self.modelGs[j].fk.from_local_to_world(self.gt_pos[j]) # / height.reshape(height.shape + (1, ))

            # rec_loss3 = self.rec_criterion(self.output_pos_global[j], self.gt_pos_global[j])
            # self.rec_losses3.append(rec_loss3.item())

            # Total loss for backward 
            loss = 1 * (rec_loss1) # + 1 * (rec_loss2 * args.lambda_global_pos) +  # rec_loss3 * self.args.lambda_position # 
            self.rec_loss.append(loss)
            self.rec_losses.append(loss.item())

    def backward(self):
        """ backward and optimize """
        for j in range(self.args.n_topology):
            generator_loss = self.rec_loss[j]
            self.optimizerGs[j].zero_grad()
            generator_loss.backward()
            self.optimizerGs[j].step()

    def bvh_writing(self, save_dir):
        """ BVH Writing """
        for j in range(self.args.n_topology):
            """ BVH Writing """ 
            if self.epoch == 0:
                self.write_bvh(save_dir, "gt", self.denorm_gt_motions[j], self.character_idx, self.motion_idx, j)
            if self.epoch != 0:
                self.write_bvh(save_dir, "output"+str(self.epoch), self.denorm_output_motions[j], self.character_idx, self.motion_idx, j)

    def save(self, path, epoch):
        try_mkdir(path)
        for i in range(self.args.n_topology):
            path_para = os.path.join(path, "Gen" + str(i) +'_' + str(epoch))
            torch.save(self.modelGs[i].state_dict(), path_para)

            path_para = os.path.join(path, "Gen" + str(i)+ '_Opti_' + str(epoch))
            torch.save(self.optimizerGs[i].state_dict(), path_para)

    def load(self, path, epoch):
        # Generator 
        for i in range(self.args.n_topology):
            path_para = os.path.join(path, "Gen" + str(i) +'_' + str(epoch))
            if not os.path.exists(path_para):
                raise Exception('Unknown loading path')
            self.modelGs[i].load_state_dict(torch.load(path_para))

            path_para = os.path.join(path, "Gen" + str(i) + '_Opti_' + str(epoch))
            if not os.path.exists(path_para):
                raise Exception('Unknown loading path')
            self.optimizerGs[i].load_state_dict(torch.load(path_para))

        # TODO: Discriminator
        print('load succeed')

    def get_curr_motion(self, iter, batch_size):
        return iter * batch_size

    def get_curr_character(self, motion_idx, num_motions):
        return int(motion_idx / num_motions)

    def write_bvh(self, save_dir, gt_or_output_epoch, motion, character_idx, motion_idx, i):
        if i == 0: 
            group = 'topology0/'
        else: 
            group = 'topology1/'

        save_dir = save_dir + group + "character{}_{}/{}/".format(character_idx, self.characters[i][character_idx], gt_or_output_epoch)
        try_mkdir(save_dir)
        file = BVH_file(option_parser.get_std_bvh(dataset=self.characters[i][character_idx]))

        bvh_writer = BVH_writer(file.edges, file.names)
        for j in range(self.args.batch_size):
            file_name = save_dir + "motion_{}.bvh".format(int(motion_idx % self.args.num_motions + j))
            bvh_writer.write_raw(motion[j], self.args.rotation, file_name)

    def denormalize(self, character_idx, motions, i):
        return self.dataset.denorm(i, character_idx, motions)