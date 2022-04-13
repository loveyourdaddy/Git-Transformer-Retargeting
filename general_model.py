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

# SAVE_ATTENTION_DIR = "attention_vis"
# SAVE_ATTENTION_DIR_INTRA = "attention_vis_intra"
# os.makedirs(SAVE_ATTENTION_DIR, exist_ok=True)
# os.makedirs(SAVE_ATTENTION_DIR_INTRA, exist_ok=True)

def try_mkdir(path):
    if not os.path.exists(path):
        # print('make new dir')
        os.system('mkdir -p {}'.format(path))

class GeneralModel():
    def __init__(self, args, character_names, dataset):
        # super(GeneralModel, self).__init__(args)
        self.args = args
        self.dataset = dataset
        self.characters = character_names
        offsets = dataset.get_offsets()
        self.n_topology = self.args.n_topology
        # self.num_bp = 6

        """ Models """ 
        self.models = []
        self.G_para = []
        self.D_para = []
        # self.optimizerDs = []

        args.input_size = dataset[0][0][0].size(0)
        args.output_size = dataset[0][1][0].size(0)

        for i in range(self.n_topology):
            model = MotionGenerator(args, offsets, i)
            model = model.to(args.cuda_device)
            self.models.append(model)
            self.G_para += self.models[i].G_parameters() # 96 for each model 
            # self.D_para += self.models[i].D_parameters()

        self.optimizerGs = torch.optim.Adam(self.G_para, lr=args.learning_rate) 
        # self.optimizerDs = torch.optim.Adam(self.D_para, lr=args.learning_rate) 

        """ Set BVH writers """ 
        self.files = []
        self.writers = []
        for i in range(len(character_names)):
            bvh_writers = []
            files = []
            for j in range(len(character_names[0])):
                file = BVH_file(option_parser.get_std_bvh(dataset=character_names[i][j]))
                files.append(file)
                bvh_writers.append(BVH_writer(file.edges, file.names))

            self.files.append(files)
            self.writers.append(bvh_writers)
    
        """ define lists"""
        self.DoF        = [] 
        self.offset_idx = [0] * self.n_topology

        # pos 
        self.output_pos         = [0] * self.n_topology
        self.gt_pos             = [0] * self.n_topology
        self.output_pos_global  = [0] * self.n_topology
        self.gt_pos_global      = [0] * self.n_topology

        """ Criternion """
        self.rec_criterion = torch.nn.MSELoss() 
        self.cycle_criterion = torch.nn.MSELoss() 
        self.gan_criterion = GAN_loss(args.gan_mode).to(args.cuda_device)

        # index
        # self.character_idx = 0 
        # self.motion_idx = 0 

        """ update input/output dimension of network: Set DoF """
        for j in range(self.n_topology):
            motion = self.dataset[0][j][0] # (256, 4, motion/offset 2, 91 or 111, 128)
            # motion = self.dataset[0][j][0][0]
            self.DoF.append(motion.size(0))

    def train_epoch(self, epoch, data_loader, save_name):
        self.epoch = epoch
        save_dir = self.args.save_dir + save_name
        try_mkdir(save_dir)

        # self.prev_setting()

        for i in range(self.n_topology):
            self.models[i].train()

        with tqdm(total=len(data_loader), desc=f"TrainEpoch {epoch}") as pbar:
            for i, value in enumerate(data_loader):
                self.iter_setting(i)
                self.separate_bp_motion(value)
                self.feed_to_network()
                # self.combine_full_motion()
                self.denorm_motion()
                self.get_loss()
                
                self.backward_G()
                if self.epoch % 100 == 0:
                    self.bvh_writing(save_dir)

                """ show info """
                pbar.update(1)
                pbar.set_postfix_str(f"element: {np.mean(self.element_losses):.3f}, cross: {np.mean(self.cross_losses):.3f}") 

    def iter_setting(self, i):
        if self.args.is_train == 1:
            self.motion_idx = self.get_curr_motion(i, self.args.batch_size)
            self.character_idx = self.get_curr_character(self.motion_idx, self.args.num_motions)
        else:  # TODO : character index for test set
            self.motion_idx = 0
            self.character_idx = 0

        self.input_motions = [] 
        self.output_motions = []
        self.gt_motions = []
        self.fake_motions = []
        
        # bp
        self.motions = []
        self.bp_motions = []
        self.bp_output_motions = []
        self.bp_fake_motions = [] 
        
        # latent codes 
        self.latents = []
        self.fake_latents = []
        self.bp_latents = []
        self.bp_fake_latents = [] 

        # denorm
        self.denorm_gt_motions      = []
        self.denorm_fake_motions    = []
        self.denorm_output_motions  = []

        # loss 1
        self.element_losses = []
        self.cross_losses = []
        self.root_losses = [] 
        self.rec_losses = [] 
        
        # loss 2 
        self.cycle_losses = [] 
        self.latent_losses = []

        # loss 3
        # self.G_losses = []
        self.G_fake_losses = []
        self.D_real_losses = []
        self.D_fake_losses = []

    def separate_bp_motion(self, value):
        for j in range(self.n_topology):
            motions, self.offset_idx[j] = value[j]
            motions = motions.to(self.args.cuda_device)
            self.gt_motions.append(motions)
            
            # self.motion_length = motions.size(2)
            # if self.args.is_train == 1:
            #     first_index = self.args.batch_size
            # else:
            #     first_index = len(self.characters[0])

            # bp_motions = torch.zeros(first_index, 6, self.DoF[j], self.motion_length).to(self.args.cuda_device)

            # """ body part motions for source motion """ # body part: (:, 0~4, 0~90, :)
            # index = self.dataset.groups_body_parts_index
            # for b, bp_index in enumerate(index[j]):
            #     bp_motions[:, b, bp_index, :] = motions[:, bp_index, :]             
            # for quat_idx in range(4): # root rotation (:, 0, 0~91, :)
            #     bp_motions[:, 5, quat_idx, :] = motions[:, quat_idx, :]
            # for quat_idx in range(3): # root position : 4~7 
            #     bp_motions[:, 5, self.DoF[j] - 3 + quat_idx, :] = motions[:, self.DoF[j] - 3 + quat_idx, :]

            # self.bp_motions.append(bp_motions)

    def feed_to_network(self):
        for j in range(self.n_topology):
            _, gt_latent = self.models[j].transformer(self.gt_motions[j])
            # bp_latent = torch.zeros(bp_latent.size(0), 6, bp_latent.size(1), bp_latent.size(2)).to(self.args.cuda_device)

            # for b in range(6):
            # _, gt_latent = self.models[j].transformer(self.gt_motions[j])

            self.latents.append(gt_latent)

        """ Get fake output and fake latent code """ 
        for src in range(self.n_topology):
            for dst in range(self.n_topology):
                # for b in range(6):
                fake_motion, _, _ = self.models[dst].transformer.decoder(self.latents[src])
                fake_latent, _, _ = self.models[dst].transformer.encoder(fake_motion)

                # fake_motion = torch.unsqueeze(fake_motion, 1)
                # fake_latent = torch.unsqueeze(fake_latent, 1)
                
                # if b == 0:
                #     bp_fake_motions = bp_fake_motion
                #     bp_fake_latents = bp_fake_latent
                # else: 
                #     bp_fake_motions = torch.cat([bp_fake_motions, bp_fake_motion], dim=1)
                #     bp_fake_latents = torch.cat([bp_fake_latents, bp_fake_latent], dim=1)

                self.fake_motions.append(fake_motion)
                self.fake_latents.append(fake_latent)
        
    # def combine_full_motion(self):
    #     """ Combine part by motion back """              
    #     index = self.dataset.groups_body_parts_index 
    #     if self.args.is_train == 1:
    #         first_index = self.args.batch_size
    #     else:
    #         first_index = len(self.characters[0])

    #     for src in range(self.n_topology):
    #         for dst in range(self.n_topology):
    #             fake_motions = torch.zeros(first_index, self.DoF[dst], self.motion_length).to(self.args.cuda_device)

    #             for b in range(5): # body parts
    #                 fake_motions[:, index[dst][b], :] = self.bp_fake_motions[2*src+dst][:, b, index[dst][b], :]
    #             for k in range(4): # root rotation (0~3)
    #                 fake_motions[:, k, :]  = self.bp_fake_motions[2*src+dst][:, 5, k, :]
    #             for k in range(3): # position (len-3 ~ -1)
    #                 fake_motions[:, self.DoF[dst] - 3 + k, :] = self.bp_fake_motions[2*src+dst][:, 5, self.DoF[dst] - 3 + k, :]
    #             self.fake_motions.append(fake_motions)

    def denorm_motion(self):
        """ Denorm and transpose & Remake root & Get global position """
        for j in range(self.n_topology):
            if self.args.normalization == 1:
                denorm_gt_motions = self.denormalize(self.character_idx, self.gt_motions[j], j)
            else:
                denorm_gt_motions = self.gt_motions[j]
            self.denorm_gt_motions.append(denorm_gt_motions)

        for src in range(self.n_topology):
            for dst in range(self.n_topology):
                if self.args.normalization == 1:
                    denorm_fake_motions = self.denormalize(self.character_idx, self.fake_motions[2*src+dst], dst)
                else:
                    denorm_fake_motions = self.fake_motions[j]
                self.denorm_fake_motions.append(denorm_fake_motions)

    def get_loss(self):
        self.G_loss = 0
        self.rec_loss = 0 
        self.latent_loss = 0 
        self.cycle_loss = 0 
        self.gan_loss = 0

        """ loss1. reconstruction loss for intra structure retargeting """
        for src in range(self.n_topology):
            # loss1-1. on each element
            element_loss = self.rec_criterion(self.gt_motions[src], self.fake_motions[3*src]) # forward: 0 / 3
            self.element_losses.append(element_loss.item())

            # loss1-2. root 
            # root_loss = self.rec_criterion(self.denorm_gt_motions[j][:, -3:, :], self.denorm_output_motions[j][:, -3:, :]) # / height
            # self.root_losses.append(root_loss.item())

            # loss 1-3. global_pos_loss

            # Total loss 
            rec_loss = (1 * element_loss) #+ (2.5 * root_loss) # + 1* global_pos_loss
            self.rec_loss += rec_loss

            self.rec_losses.append(rec_loss.item())

        """ 2. latent consisteny and cycle loss for intra and cross strucuture retargeting  """ 
        latent_loss = self.cycle_criterion(self.latents[0], self.latents[1])
        self.latent_loss += latent_loss
        self.latent_losses.append(latent_loss.item())
        
        for src in range(self.n_topology):
            for dst in range(self.n_topology):
                cycle_loss = self.cycle_criterion(self.latents[dst], self.fake_latents[2*src+dst])
                self.cycle_loss += cycle_loss
                self.cycle_losses.append(cycle_loss.item())
        
        # """ 3. GAN loss for each body part """
        # p = 0 
        # for src in range(self.n_topology):
        #     for dst in range(self.n_topology):
        #         for b in range(6):
        #             netD = self.models[dst].body_part_discriminator[b]

        #             fake_pred = netD(self.fake_motions[2*src+dst])
        #             G_fake_loss = self.gan_criterion(fake_pred, True)
        #             self.gan_loss += G_fake_loss
        #             self.G_fake_losses.append(G_fake_loss.item())

        self.G_loss = (self.rec_loss) + (self.latent_loss) # + self.cycle_loss # + self.gan_loss

        # cross loss
        cross_loss = self.rec_criterion(self.fake_motions[1], self.gt_motions[1]) # src 0->dst 1
        self.cross_losses.append(cross_loss.item())
        cross_loss = self.rec_criterion(self.fake_motions[2], self.gt_motions[0]) # src 1->dst 0
        self.cross_losses.append(cross_loss.item())

    def backward_G(self):
        """ backward and optimize """
        self.optimizerGs.zero_grad()
        self.G_loss.backward()
        self.optimizerGs.step()

    def backward_D(self):
        self.D_loss = 0
        for src in range(self.args.n_topology):
            for dst in range(self.args.n_topology):
                for b in range(6):
                    netD = self.models[dst].body_part_discriminator[b]

                    # output of real motion
                    real_pred = netD(self.gt_motions[dst])
                    D_real_loss = self.gan_criterion(real_pred, True)
                    self.D_real_losses.append(D_real_loss.item())

                    # output of fake motion
                    fake_pred = netD(self.fake_motions[2*src+dst].detach())
                    D_fake_loss = self.gan_criterion(fake_pred, False)
                    self.D_fake_losses.append(D_fake_loss.item())

                    D_loss = (D_real_loss + D_fake_loss) * 0.5
                    self.D_loss += D_loss

        self.D_loss.backward()
        self.optimizerDs.step()

    def save(self, path, epoch):
        for i, model in enumerate(self.models):
            file_name = os.path.join(path, 'topology{}'.format(i), 'epoch{}.pt'.format(epoch))
            try_mkdir(os.path.split(file_name)[0])
            # for b, bp_generator in enumerate(model.generators):
            torch.save(model.state_dict(), file_name)

        file_name = os.path.join(path, 'optimizer', 'epoch{}.pt'.format(epoch))
        try_mkdir(os.path.split(file_name)[0])
        torch.save(self.optimizerGs.state_dict(), file_name)

    def load(self, path, epoch):
        # Generator 
        # model 
        for i, model in enumerate(self.models):
            file_name = os.path.join(path, 'topology{}'.format(i), 'epoch{}.pt'.format(epoch))
            # for b, bp_generator in enumerate(model.bp_generators):
                # torch.load(bp_generator.load)
            # path_to_load = os.path.join(file_name, 'bp{}.pt'.format(b))
            model.load_state_dict(torch.load(file_name, map_location=self.args.cuda_device))
     
        # optimizer 
        file_name = os.path.join(path, 'optimizer', 'epoch{}.pt'.format(epoch))
        self.optimizerGs.load_state_dict(torch.load(file_name))
      
        # TODO: Discriminator
        print('load succeed')

    def get_curr_motion(self, iter, batch_size):
        return iter * batch_size

    def get_curr_character(self, motion_idx, num_motions):
        return int(motion_idx / num_motions)

    def bvh_writing(self, save_dir): # for training 
        """ BVH Writing """
        if self.epoch == 0: 
            for j in range(self.n_topology):
                self.write_bvh(save_dir, "gt", self.denorm_gt_motions[j], self.character_idx, self.motion_idx, j)

        for src in range(self.n_topology):
            # self.write_bvh(save_dir, "fake"+str(self.epoch)+"_"+str(src), self.denorm_fake_motions[src], self.character_idx, self.motion_idx, src)
            for dst in range(self.n_topology):
                self.write_bvh(save_dir, "fake"+str(self.epoch)+"_"+str(src)+"_"+str(dst), self.denorm_fake_motions[2*src+dst], self.character_idx, self.motion_idx, dst)

    def write_bvh(self, save_dir, gt_or_output_epoch, motion, character_idx, motion_idx, i):
        if i == 0: 
            group = 'topology0/'
        else: 
            group = 'topology1/'

        save_dir = save_dir + group + "character{}_{}/{}/".format(character_idx, self.characters[i][character_idx], gt_or_output_epoch)
        try_mkdir(save_dir)

        for j in range(self.args.batch_size):
            if gt_or_output_epoch == 'gt':
                file_name = save_dir + "gt_{}.bvh".format(int(motion_idx % self.args.num_motions + j))
            else :
                file_name = save_dir + "motion_{}.bvh".format(int(motion_idx % self.args.num_motions + j))
            self.writers[i][character_idx].write_raw(motion[j], self.args.rotation, file_name)

    def denormalize(self, character_idx, motions, i):
        return self.dataset.denorm(i, character_idx, motions)

    def remake_root_position_from_displacement(self, motions, num_bs, num_frame, num_DoF):
        for bs in range(num_bs):  # dim 0
            for frame in range(num_frame - 1):  # dim 2 # frame: 0~62. update 1 ~ 63
                motions[bs][frame + 1][num_DoF - 3] += motions[bs][frame][num_DoF - 3]
                motions[bs][frame + 1][num_DoF - 2] += motions[bs][frame][num_DoF - 2]
                motions[bs][frame + 1][num_DoF - 1] += motions[bs][frame][num_DoF - 1]

        return motions

    # def discriminator_requires_grad_(self, requires_grad):
    #     for model in self.models:
    #         for b in range(6):
    #             # for para in model.body_part_discriminator[b].parameters(): # self.models[dst].body_part_discriminator[b] # D_parameters
    #             for para in model.D_parameters():
    #                 para.requires_grad = requires_grad

    """ eval """
    def eval_epoch(self, epoch, dataset, save_name):
        save_dir = self.args.save_dir + save_name +'test/'
        try_mkdir(save_dir)

        for i in range(self.n_topology):
            self.models[i].eval()
            motion = self.dataset[0][i][0]
            self.DoF[i] = motion.size(1)

        self.id_test = 0
        with tqdm(total=len(dataset), desc=f"TestEpoch {epoch}") as pbar:
            for i, value in tqdm(enumerate(dataset)):
                self.iter_setting(i)

                self.separate_bp_motion(value)
                self.feed_to_network_test(value)
                self.combine_full_motion()
                self.compute_test_result(save_dir)

                pbar.update(1)
                pbar.set_postfix_str(f"element: {np.mean(self.element_losses):.3f}, cross: {np.mean(self.cross_losses):.3f}") 

    def feed_to_network_test(self, motions):
        for j in range(self.n_topology):
            motion, self.offset_idx[j] = motions[j]
            motion = motion.to(self.args.cuda_device)

        for j in range(self.n_topology):
            bp_output_motions = torch.zeros(self.bp_motions[j].size()).to(self.args.cuda_device)
            for b in range(6):
                bp_output_motions[:,b,:,:], bp_latent = self.models[j].bp_generators[b](self.bp_motions[j][:,b,:,:])
                bp_latent = torch.unsqueeze(bp_latent, 1)

                if b == 0:
                    bp_latents = bp_latent
                else:
                    bp_latents = torch.cat([bp_latents, bp_latent], dim=1)

            self.bp_output_motions.append(bp_output_motions)
            self.bp_latents.append(bp_latents)

        """ Get fake output and fake latent code """ 
        for src in range(self.n_topology):
            for dst in range(self.n_topology):
                for b in range(6):
                    bp_fake_motion = self.models[dst].bp_generators[b].decoder(self.bp_latents[src][:,b,:,:])
                    bp_fake_latent = self.models[dst].bp_generators[b].encoder(bp_fake_motion)
                    
                    bp_fake_motion = torch.unsqueeze(bp_fake_motion, 1)
                    bp_fake_latent = torch.unsqueeze(bp_fake_latent, 1)
                    
                    if b == 0:
                        bp_fake_motions = bp_fake_motion
                        bp_fake_latents = bp_fake_latent
                    else: 
                        bp_fake_motions = torch.cat([bp_fake_motions, bp_fake_motion], dim=1)
                        bp_fake_latents = torch.cat([bp_fake_latents, bp_fake_latent], dim=1)

                self.bp_fake_motions.append(bp_fake_motions)
                self.bp_fake_latents.append(bp_fake_latents)

    def compute_test_result(self, save_dir):
        for src in range(self.n_topology):
            gt = self.gt_motions[src]
            idx = list(range(gt.shape[0]))
            gt = self.dataset.denorm(src, idx, gt) # 여기가 잘못되었을 가능성이 큼
            for i in idx: # i = [0,1,2,3]
                new_path = os.path.join(save_dir, self.characters[src][i])
                try_mkdir(new_path)
                self.writers[src][i].write_raw(gt[i, ...], 'quaternion', 
                                            os.path.join(new_path, '{}_gt.bvh'.format(self.id_test)))
        for src in range(self.n_topology):
            for dst in range(self.n_topology):
                output = self.fake_motions[2*src+dst]
                idx = list(range(output.shape[0]))
                output = self.dataset.denorm(dst, idx, output)
                for i in idx: # i = [0,1,2,3]
                    new_path = os.path.join(save_dir, self.characters[src][i])
                    try_mkdir(new_path)
                    self.writers[dst][i].write_raw(output[i, ...], 'quaternion', 
                                            os.path.join(new_path, '{}_output_{}_{}.bvh'.format(self.id_test, src, dst)))

        self.id_test += 1 
