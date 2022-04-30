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
from models.utils import GAN_loss, get_ee, Criterion_EE
from model import MotionGenerator

def try_mkdir(path):
    if not os.path.exists(path):
        os.system('mkdir -p {}'.format(path))


class GeneralModel():
    def __init__(self, args, character_names, dataset):
        self.args = args
        self.dataset = dataset
        self.characters = character_names
        offsets = dataset.get_offsets()
        self.n_topology = self.args.n_topology

        """ update input/output dimension of network: Set DoF """
        self.DoF = []
        for j in range(self.n_topology):
            # (256 total motions , 4 characters, 2 motion/offset, DoF, window)
            motion = self.dataset[0][j][0]
            self.DoF.append(motion.size(0))
        args.input_size = dataset[0][0][0].size(0)
        args.output_size = dataset[0][1][0].size(0)

        """ Models """
        self.models = []
        self.G_para = []
        self.D_para = []

        for i in range(self.n_topology):
            model = MotionGenerator(args, offsets, i)
            model = model.to(args.cuda_device)
            self.models.append(model)
            self.G_para += self.models[i].G_parameters()
            self.D_para += self.models[i].D_parameters()

        self.optimizerGs = torch.optim.Adam(self.G_para, lr=args.learning_rate)
        self.optimizerDs = torch.optim.Adam(self.D_para, lr=args.learning_rate)

        """ Set BVH writers """
        self.files = []
        self.writers = []
        self.FKs = []
        self.height = []
        for i in range(len(character_names)):
            bvh_writers = []
            files = []
            FKs = []
            height = []
            for j in range(len(character_names[0])):
                file = BVH_file(option_parser.get_std_bvh(
                    dataset=character_names[i][j]))
                files.append(file)
                bvh_writers.append(BVH_writer(file.edges, file.names))
                FKs.append(ForwardKinematics(args, file.edges))
                height.append(file.get_height())
            self.height.append(height)
            self.files.append(files)
            self.writers.append(bvh_writers)
            self.FKs.append(FKs)

        """ define lists"""
        self.offset_idx = [0] * self.n_topology

        """ Criternion """
        self.rec_criterion = torch.nn.MSELoss()
        self.cycle_criterion = torch.nn.MSELoss()
        self.gan_criterion = GAN_loss(args.gan_mode).to(args.cuda_device)
        self.criterion_ee = Criterion_EE(args, torch.nn.MSELoss())

    def train_epoch(self, epoch, data_loader, save_name):
        self.epoch = epoch
        save_dir = self.args.save_dir + save_name
        try_mkdir(save_dir)

        # self.prev_setting()

        for i in range(self.n_topology):
            self.models[i].train()

        with tqdm(total=len(data_loader), desc=f"TrainEpoch {epoch}") as pbar:
            self.iter_setting()
            for i, value in enumerate(data_loader):
                self.motion_idx = self.get_curr_motion(i, self.args.batch_size)
                self.character_idx = self.get_curr_character(
                    self.motion_idx, self.args.num_motions)

                self.separate_motion(value)
                self.feed_to_network()
                self.denorm_motion()
                self.get_loss()

                self.discriminator_requires_grad_(False)
                self.backward_G()

                self.discriminator_requires_grad_(True)
                self.backward_D()

                if self.epoch % self.args.save_epoch == 0:
                    self.bvh_writing(save_dir)

                """ show info """
                pbar.update(1)
                pbar.set_postfix_str(
                    f"element: {np.mean(self.element_losses):.3f}, cross: {np.mean(self.cross_losses):.3f}, fk: {np.mean(self.fk_losses):.3f}")

    def iter_setting(self):
        # loss 1
        self.element_losses = []
        self.cross_losses = []
        self.fk_losses = []
        self.smooth_losses = []
        self.root_losses = []
        self.root_rotation_losses = []

        # loss 2
        self.cycle_losses = []
        self.latent_losses = []

        # loss 3
        self.ee_losses = []

        # loss 4
        self.G_fake_losses = []
        self.D_real_losses = []
        self.D_fake_losses = []

    def separate_motion(self, value):
        # dataset
        self.gt_motions = []
        for j in range(self.n_topology):
            motions, self.offset_idx[j] = value[j]
            # (bs,DoF,window)->(window,bs,DoF)
            motions = motions.permute(2, 0, 1)
            motions = motions.to(self.args.cuda_device)
            self.gt_motions.append(motions)

    def feed_to_network(self):
        self.output_motions = []
        self.latents = []
        self.fake_motions = []
        self.fake_latents = []

        for src in range(self.n_topology):
            # reconstruction: output
            output_motion, latent = self.models[src].transformer.forward(
                self.gt_motions[src], self.gt_motions[src]  # src, tgt
            )
            self.output_motions.append(output_motion)
            self.latents.append(latent)

            # retargeting: fake
            for dst in range(self.n_topology):
                fake_motion = self.models[dst].transformer.dec_forward(
                    self.gt_motions[dst], latent
                )
                fake_latent = self.models[dst].transformer.enc_forward(
                    fake_motion
                )
                self.fake_motions.append(fake_motion)
                self.fake_latents.append(fake_latent)

    def denorm_motion(self):
        """ Denorm and transpose & Remake root & Get global position """
        self.denorm_gt_motions = []
        self.denorm_output_motions = []
        self.denorm_fake_motions = []
        for src in range(self.n_topology):
            # reconstruction
            # (window,bs,DoF)->(bs,DoF,window)
            gt_motions = self.gt_motions[src].permute(1, 2, 0)
            output_motions = self.output_motions[src].permute(1, 2, 0)

            if self.args.normalization == 1:
                denorm_gt_motions = self.denormalize(
                    self.character_idx, gt_motions, src)
                denorm_output_motions = self.denormalize(
                    self.character_idx, output_motions, src)
            else:
                denorm_gt_motions = gt_motions
                denorm_output_motions = output_motions

            if self.args.root_pos_as_disp == 1:
                denorm_gt_motions = self.root_displacement_to_position(
                    denorm_gt_motions, src)
                denorm_output_motions = self.root_displacement_to_position(
                    denorm_output_motions, src)

            self.denorm_gt_motions.append(denorm_gt_motions)
            self.denorm_output_motions.append(denorm_output_motions)

            # retargeting
            for dst in range(self.n_topology):
                idx = self.n_topology * src + dst
                fake_motion = self.fake_motions[idx].permute(
                    1, 2, 0)  # from src to dst
                if self.args.normalization == 1:
                    denorm_fake_motions = self.denormalize(
                        self.character_idx, fake_motion, dst)
                else:
                    denorm_fake_motions = fake_motion

                if self.args.root_pos_as_disp == 1:
                    denorm_fake_motions = self.root_displacement_to_position(
                        denorm_fake_motions, dst)
                self.denorm_fake_motions.append(denorm_fake_motions)

        # save fk and ee
        self.gt_pos = []
        self.output_pos = []
        self.fake_pos = []

        self.gt_global_pos = []
        self.output_global_pos = []
        self.fake_global_pos = []

        self.gt_ee = []
        self.output_ee = []
        self.fake_ee = []
        
        for src in range(self.n_topology):
            offset = self.dataset.offsets_group[src][self.character_idx]
            fk = self.FKs[src][self.character_idx]

            gt_pos = fk.forward_from_raw(
                self.denorm_gt_motions[src], offset)
            output_pos = fk.forward_from_raw(
                self.denorm_output_motions[src], offset)
            self.gt_pos.append(gt_pos)
            self.output_pos.append(output_pos)

            gt_global_pos = fk.from_local_to_world(gt_pos)
            output_global_pos = fk.from_local_to_world(output_pos)

            self.gt_global_pos.append(gt_global_pos)
            self.output_global_pos.append(output_global_pos)

            gt_ee = get_ee(gt_pos, self.dataset.joint_topologies[src], self.dataset.ee_ids[src],
                           velo=self.args.ee_velo, from_root=self.args.ee_from_root)
            output_ee = get_ee(output_pos, self.dataset.joint_topologies[src], self.dataset.ee_ids[src],
                               velo=self.args.ee_velo, from_root=self.args.ee_from_root)
            self.gt_ee.append(gt_ee)
            self.output_ee.append(output_ee)

            # retargeting
            for dst in range(self.n_topology):
                offset = self.dataset.offsets_group[dst][self.character_idx]
                fk = self.FKs[dst][self.character_idx]

                idx = self.n_topology * src + dst

                fake_pos = fk.forward_from_raw(
                    self.denorm_fake_motions[idx], offset)  # from src
                self.fake_pos.append(fake_pos)

                fake_global_pos = fk.from_local_to_world(fake_pos)
                self.fake_global_pos.append(fake_global_pos)

                fake_ee = get_ee(fake_pos, self.dataset.joint_topologies[dst], self.dataset.ee_ids[dst],
                                 velo=self.args.ee_velo, from_root=self.args.ee_from_root)
                self.fake_ee.append(fake_ee)

    def get_loss(self):
        """ loss1. reconstruction loss for intra structure retargeting """
        self.rec_loss = 0
        self.root_loss = 0
        self.fk_loss = 0
        for src in range(self.n_topology):
            # loss1-1. on each element 1``
            element_loss = self.rec_criterion(
                self.gt_motions[src], self.fake_motions[3*src]
            )
            self.rec_loss += element_loss
            self.element_losses.append(element_loss.item())

            # loss1-2. root
            height = self.height[src][self.character_idx]
            root_loss = self.rec_criterion(
                self.gt_motions[src][:, :, -3:] / height,
                self.fake_motions[3*src][:, :, -3:] / height
            )
            self.root_loss += root_loss
            self.root_losses.append(root_loss.item())

            # loss 1-3. fk
            fk_loss = self.rec_criterion(
                self.gt_global_pos[src], self.output_global_pos[src]
            )
            self.fk_loss += fk_loss
            self.fk_losses.append(fk_loss.item())

            # check: smooth on next frame
            smooth_loss = self.rec_criterion(
                self.fake_motions[3*src][:-1, :, :],
                self.fake_motions[3*src][1:, :, :]
            )
            self.smooth_losses.append(smooth_loss.item())

            # check: root roation
            root_rotation_loss = self.rec_criterion(
                self.gt_motions[src][:, :, :4],
                self.fake_motions[3*src][:, :, :4]
            )
            self.root_rotation_losses.append(root_rotation_loss.item())

        """ loss 2. latent consisteny and cycle loss for intra and cross strucuture retargeting  """
        # loss 2-1. cycle loss
        self.cycle_loss = 0
        for src in range(self.n_topology):
            for dst in range(self.n_topology):
                idx = self.n_topology * src + dst
                cycle_loss = self.cycle_criterion(
                    self.latents[dst], self.fake_latents[idx])
                self.cycle_loss += cycle_loss
                self.cycle_losses.append(cycle_loss.item())

        # loss 2-2. check: common latent loss
        latent_loss = self.cycle_criterion(self.latents[0], self.latents[1])
        self.latent_losses.append(latent_loss.item())

        """ loss 3. ee loss """
        self.ee_loss = 0
        for src in range(self.n_topology):
            for dst in range(self.n_topology):
                idx = self.n_topology * src + dst
                ee_loss = self.criterion_ee(self.gt_ee[dst], self.fake_ee[idx])
                self.ee_loss += ee_loss
                self.ee_losses.append(ee_loss.item())

        """ loss 4. GAN loss """
        self.gan_loss = 0
        for src in range(self.n_topology):
            for dst in range(self.n_topology):
                netD = self.models[dst].discriminator

                fake_pred = netD(
                    self.gt_motions[dst], self.fake_motions[2*src+dst])
                G_fake_loss = self.gan_criterion(fake_pred, True)
                self.gan_loss += G_fake_loss
                self.G_fake_losses.append(G_fake_loss.item())

        self.G_loss = 5*(self.rec_loss) + 1250*(self.root_loss) + 500*(self.fk_loss)\
            + 5 * (self.cycle_loss) \
            + 100 * (self.ee_loss) \
            + (self.gan_loss)

        # cross loss
        cross_loss = self.rec_criterion(
            self.fake_motions[1], self.gt_motions[1])  # src 0 -> dst 1
        self.cross_losses.append(cross_loss.item())
        cross_loss = self.rec_criterion(
            self.fake_motions[2], self.gt_motions[0])  # src 1 -> dst 0
        self.cross_losses.append(cross_loss.item())

    def bvh_writing(self, save_dir):  # for training
        """ BVH Writing """
        if self.epoch == 0:
            for j in range(self.n_topology):
                self.write_bvh(save_dir, "gt",
                               self.denorm_gt_motions[j], self.character_idx, self.motion_idx, j)

        for src in range(self.n_topology):
            for dst in range(self.n_topology):
                self.write_bvh(save_dir, "fake"+str(self.epoch)+"_"+str(src)+"_"+str(dst),
                               self.denorm_fake_motions[2*src+dst], self.character_idx, self.motion_idx, dst)

    def write_bvh(self, save_dir, gt_or_output_epoch, motion, character_idx, motion_idx, i):
        if i == 0:
            group = 'topology0/'
        else:
            group = 'topology1/'

        save_dir = save_dir + group + "character{}_{}/{}/".format(
            character_idx, self.characters[i][character_idx], gt_or_output_epoch)
        try_mkdir(save_dir)

        for j in range(self.args.batch_size):
            if gt_or_output_epoch == 'gt':
                file_name = save_dir + \
                    "gt_{}.bvh".format(
                        int(motion_idx % self.args.num_motions + j))
            else:
                file_name = save_dir + \
                    "motion_{}.bvh".format(
                        int(motion_idx % self.args.num_motions + j))

            self.writers[i][character_idx].write_raw(
                motion[j], self.args.rotation, file_name)

    def denormalize(self, character_idx, motions, i):
        return self.dataset.denorm(i, character_idx, motions)

    def root_displacement_to_position(self, motions, j):
        if j == 0:
            num_DoF = self.args.input_size
        else:
            num_DoF = self.args.output_size

        motion_len = motions.shape[2]

        for frame in range(motion_len - 1):
            motions[:, num_DoF-3:, frame + 1] += motions[:, num_DoF-3:, frame]

        return motions

    def backward_G(self):
        """ backward and optimize """
        self.optimizerGs.zero_grad()
        self.G_loss.backward()
        self.optimizerGs.step()

    def backward_D(self):
        self.optimizerDs.zero_grad()

        self.D_loss = 0
        for src in range(self.args.n_topology):
            # output of real motion
            netD = self.models[src].discriminator
            real_pred = netD(self.gt_motions[src], self.gt_motions[src])
            D_real_loss = self.gan_criterion(real_pred, True)
            self.D_real_losses.append(D_real_loss.item())

            for dst in range(self.args.n_topology):
                netD = self.models[dst].discriminator

                # output of fake motion
                fake_pred = netD(
                    self.fake_motions[2*src+dst].detach(), self.gt_motions[dst])
                D_fake_loss = self.gan_criterion(fake_pred, False)
                self.D_fake_losses.append(D_fake_loss.item())

                D_loss = (D_real_loss + D_fake_loss) * 0.5
                self.D_loss += D_loss

        self.D_loss.backward()
        self.optimizerDs.step()

    def discriminator_requires_grad_(self, requires_grad):
        for model in self.models:
            for para in model.D_parameters():
                para.requires_grad = requires_grad

    """ eval """

    def eval_epoch(self, epoch, loader, save_name):
        save_dir = self.args.save_dir + save_name + 'test/'
        try_mkdir(save_dir)

        for i in range(self.n_topology):
            self.models[i].eval()
            motion = self.dataset[0][i][0]
            self.DoF[i] = motion.size(1)

        self.id_test = 0
        with tqdm(total=len(loader), desc=f"TestEpoch {epoch}") as pbar:
            with torch.no_grad():
                for i, value in enumerate(loader):
                    self.iter_setting(i)
                    self.separate_motion_test(value)
                    self.feed_to_network_test()
                    self.denorm_motion()
                    self.get_loss()
                    self.compute_test_result(save_dir)

                    pbar.update(1)
                    pbar.set_postfix_str(
                        f"element: {np.mean(self.element_losses):.3f}, cross: {np.mean(self.cross_losses):.3f}, fk: {np.mean(self.fk_losses):.3f}")

    def separate_motion_test(self, value):
        # dataset
        for j in range(self.n_topology):
            motions, self.offset_idx[j] = value[j]
            # (bs,DoF,window)->(window,bs,DoF)
            motions = motions.permute(2, 0, 1)
            motions = motions.to(self.args.cuda_device)
            self.gt_motions.append(motions)

    def feed_to_network_test(self):
        """ Get fake output and fake latent code """
        for src in range(self.n_topology):
            latents = self.models[src].transformer.enc_forward(
                self.gt_motions[src])
            self.latents.append(latents)
            for dst in range(self.n_topology):

                fake_motions = self.models[dst].transformer.infer_dec_forward(
                    self.gt_motions[dst], latents)
                fake_latents = self.models[dst].transformer.enc_forward(
                    fake_motions)

                self.fake_motions.append(fake_motions)
                self.fake_latents.append(fake_latents)

    def compute_test_result(self, save_dir):
        for src in range(self.n_topology):
            gt = self.denorm_gt_motions[src]
            idx = list(range(gt.shape[0]))
            for i in idx:  # i = [0,1,2,3]
                new_path = os.path.join(save_dir, self.characters[src][i])
                try_mkdir(new_path)
                self.writers[src][i].write_raw(gt[i, ...], 'quaternion',
                                               os.path.join(new_path, '{}_gt.bvh'.format(self.id_test)))
        for src in range(self.n_topology):
            for dst in range(self.n_topology):
                output = self.denorm_fake_motions[2*src+dst]
                idx = list(range(output.shape[0]))
                for i in idx:  # i = [0,1,2,3]
                    new_path = os.path.join(save_dir, self.characters[src][i])
                    try_mkdir(new_path)
                    self.writers[dst][i].write_raw(output[i, ...], 'quaternion',
                                                   os.path.join(new_path, '{}_output_{}_{}.bvh'.format(self.id_test, src, dst)))

        self.id_test += 1

    """ save and load """

    def save(self, path, epoch):
        for i, model in enumerate(self.models):
            file_name = os.path.join(
                path, 'topology{}'.format(i), 'epoch{}.pt'.format(epoch))
            try_mkdir(os.path.split(file_name)[0])
            torch.save(model.state_dict(), file_name)

        file_name = os.path.join(path, 'optimizer', 'epoch{}.pt'.format(epoch))
        try_mkdir(os.path.split(file_name)[0])
        torch.save(self.optimizerGs.state_dict(), file_name)

    def load(self, path, epoch):
        # Generator
        # model
        for i, model in enumerate(self.models):
            file_name = os.path.join(
                path, 'topology{}'.format(i), 'epoch{}.pt'.format(epoch))
            model.load_state_dict(torch.load(
                file_name, map_location=self.args.cuda_device))

        # optimizer
        file_name = os.path.join(path, 'optimizer', 'epoch{}.pt'.format(epoch))
        self.optimizerGs.load_state_dict(torch.load(file_name))

        # TODO: Discriminator
        print('load succeed')

    def get_curr_motion(self, iter, batch_size):
        return iter * batch_size

    def get_curr_character(self, motion_idx, num_motions):
        return int(motion_idx / num_motions)
