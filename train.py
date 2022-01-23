import torch
import os
import numpy as np
from wandb import set_trace
from datasets import get_character_names
import option_parser
from tqdm import tqdm
from datasets.bvh_parser import BVH_file
from datasets.bvh_writer import BVH_writer
from models.Kinematics import ForwardKinematics
from rendering import *
import torchvision
from models.utils import GAN_loss
import wandb

SAVE_ATTENTION_DIR = "attention_vis_intra"
os.makedirs(SAVE_ATTENTION_DIR, exist_ok=True)

# def get_data_numbers(motion):
#     return motion.size(0), motion.size(1), motion.size(2)


def get_curr_motion(iter, batch_size):
    return iter * batch_size


def get_curr_character(motion_idx, num_motions):
    return int(motion_idx / num_motions)


def denormalize(dataset, character_idx, motions):
    return dataset.denorm(1, character_idx, motions)


def remake_root_position_from_displacement(args, motions, num_bs, num_frame, num_DoF):
    for bs in range(num_bs):  # dim 0
        for frame in range(num_frame - 1):  # dim 2 # frame: 0~62. update 1 ~ 63
            motions[bs][frame + 1][num_DoF -
                                   3] += motions[bs][frame][num_DoF - 3]
            motions[bs][frame + 1][num_DoF -
                                   2] += motions[bs][frame][num_DoF - 2]
            motions[bs][frame + 1][num_DoF -
                                   1] += motions[bs][frame][num_DoF - 1]

    return motions


def write_bvh(save_dir, gt_or_output_epoch, motion, characters, character_idx, motion_idx, args):
    save_dir_gt = save_dir + "character{}_{}/{}/".format(
        character_idx, characters[1][character_idx], gt_or_output_epoch)
    try_mkdir(save_dir_gt)
    file = BVH_file(option_parser.get_std_bvh(
        dataset=characters[1][character_idx]))
    bvh_writer = BVH_writer(file.edges, file.names)
    for j in range(args.batch_size):
        file_name = save_dir_gt + \
            "motion_{}.bvh".format(int(motion_idx % args.num_motions + j))
        bvh_writer.write_raw(motion[j], args.rotation, file_name)


def try_mkdir(path):
    if not os.path.exists(path):
        # print('make new dir')
        os.system('mkdir -p {}'.format(path))


def train_epoch(args, epoch, modelG, modelD, optimizerG, optimizerD, train_loader, train_dataset, characters, save_name, Files):
    losses = []  # losses for 1 epoch (for all motion, all batch_size)
    fk_losses = []
    reg_losses = []
    rec_losses = []
    G_losses = []
    D_losses = []

    modelG.train()
    modelD.train()

    args.epoch = epoch
    character_idx = 0
    rec_criterion = torch.nn.MSELoss()
    gan_criterion = GAN_loss(args.gan_mode).to(args.cuda_device)

    with tqdm(total=len(train_loader), desc=f"TrainEpoch {epoch}") as pbar:
        save_dir = args.save_dir + save_name
        try_mkdir(save_dir)

        for i, value in enumerate(train_loader):
            optimizerG.zero_grad()
            # optimizerD.zero_grad()

            """ Get Data and Set value to model and Get output """
            enc_inputs, dec_inputs, gt_motions = map(
                lambda v: v.to(args.cuda_device), value)

            # """ Get Data numbers: (bs, DoF, window) """
            num_bs, Dim1, Dim2 = gt_motions.size(
                0), gt_motions.size(1), gt_motions.size(2)
            if args.swap_dim == 0:
                num_frame, num_DoF = Dim1, Dim2
            else:
                num_DoF, num_frame = Dim1, Dim2

            motion_idx = get_curr_motion(i, args.batch_size)
            character_idx = get_curr_character(motion_idx, args.num_motions)
            # file = Files[1][character_idx]
            # height = file.get_height()

            """ feed to NETWORK """
            output_motions, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs = modelG(
                character_idx, character_idx, enc_inputs, dec_inputs)

            """ save attention map """
            if epoch % 10 == 0:
                bs = enc_self_attn_probs[0].size(0)
                img_size = enc_self_attn_probs[0].size(2)
                for att_layer_index, enc_self_attn_prob in enumerate(enc_self_attn_probs):
                    att_map = enc_self_attn_prob.view(bs*4,-1,img_size,img_size)
                    torchvision.utils.save_image(att_map, f"./{SAVE_ATTENTION_DIR}/enc_{att_layer_index}_{epoch:04d}.jpg",range=(0,1), normalize=True)

                img_size = dec_self_attn_probs[0].size(2)
                for att_layer_index, dec_self_attn_prob in enumerate(dec_self_attn_probs):
                    att_map = dec_self_attn_prob.view(bs*4,-1,img_size,img_size)
                    torchvision.utils.save_image(att_map, f"./{SAVE_ATTENTION_DIR}/dec_{att_layer_index}_{epoch:04d}.jpg",range=(0,1), normalize=True)

                img_size = dec_enc_attn_probs[0].size(2)
                for att_layer_index, dec_enc_attn_prob in enumerate(dec_enc_attn_probs):
                    att_map = dec_enc_attn_prob.view(bs*4,-1,img_size,img_size)
                    torchvision.utils.save_image(att_map, f"./{SAVE_ATTENTION_DIR}/enc_dec_{att_layer_index}_{epoch:04d}.jpg",range=(0,1), normalize=True)

            """ Get LOSS (orienation & FK & regularization) """
            """ 2. atten score loss """
            if args.reg_loss == 1:
                n_layer = len(enc_self_attn_probs)
                size = enc_self_attn_probs[0].size()
                n_batch, n_heads, window_size, window_size = size

                reg_weight = args.reg_weight
                zero_tensor = torch.zeros(n_batch, n_heads, window_size, window_size, device=args.cuda_device)
                for l in range(n_layer):
                    loss = reg_weight * rec_criterion(enc_self_attn_probs[l], zero_tensor)
                    loss_sum += loss
                    reg_losses.append(loss.item())

                    loss = reg_weight * rec_criterion(dec_self_attn_probs[l], zero_tensor)
                    loss_sum += loss
                    reg_losses.append(loss.item())

                    loss = reg_weight * rec_criterion(dec_enc_attn_probs[l], zero_tensor)
                    loss_sum += loss
                    reg_losses.append(loss.item())

            """ loss1. loss on each element """
            loss_sum = 0
            if args.rec_loss == 1:
                for m in range(num_bs):
                    for j in range(num_DoF):
                        loss = rec_criterion(gt_motions, output_motions)
                        loss_sum += loss
                        # append
                        losses.append(loss.item())
                        rec_losses.append(loss.item())
                        # opt
                        # optimizerG.zero_grad()
                        # loss.backward(retain_graph=True)
                        # optimizerG.step()

            """ loss2. GAN Loss"""
            # discriminator : (fake output: 0), (real_data: 1)
            if args.gan_loss == 1:
                # update generator
                for para in modelD.parameters():
                    para.requires_grad = False
                fake_output = modelD(
                    character_idx, character_idx, output_motions, output_motions)

                for m in range(num_bs):
                    for j in range(num_DoF): 
                        G_loss = gan_criterion(fake_output, True)
                        loss_sum += G_loss
                        # append
                        G_losses.append(G_loss.item())
                        losses.append(G_loss.item())
                        # opti
                        # optimizerG.zero_grad()
                        # G_loss.backward(retain_graph=True)
                        # optimizerG.step()

                # update discriminator
                for para in modelD.parameters():
                    para.requires_grad = True
                real_output = modelD(
                    character_idx, character_idx, enc_inputs, enc_inputs)
                fake_output = modelD(
                    character_idx, character_idx, output_motions.detach(), output_motions.detach())

                # pose 단위로 discriminate을 할 수는 없나?
                for m in range(num_bs):
                    for j in range(num_DoF): 
                        D_loss = 1/2 * (gan_criterion(real_output, True) +
                                        gan_criterion(fake_output, False))
                        loss_sum += D_loss
                        # append
                        D_losses.append(D_loss.item())
                        losses.append(D_loss.item())
                        # opt
            
            # optimizerG.zero_grad()
            loss_sum.backward()
            optimizerG.step()

            # optimizerD.zero_grad()
            # loss.backward()
            optimizerD.step()

            """ 1) denorm for bvh_writing """
            if args.normalization == 1:
                denorm_gt_motions = denormalize(
                    train_dataset, character_idx, gt_motions)
                denorm_output_motions = denormalize(
                    train_dataset, character_idx, output_motions)
            else:
                denorm_gt_motions = gt_motions
                denorm_output_motions = output_motions

            """ 2) Swap output motion """
            if args.swap_dim == 1:
                gt_motions = torch.transpose(gt_motions, 1, 2)
                output_motions = torch.transpose(output_motions, 1, 2)

                denorm_gt_motions = torch.transpose(denorm_gt_motions, 1, 2)
                denorm_output_motions = torch.transpose(
                    denorm_output_motions, 1, 2)

            """ 3) remake root position from displacement """
            if args.root_pos_disp == 1:
                denorm_gt_motions = remake_root_position_from_displacement(
                    args, denorm_gt_motions, num_bs, num_frame, num_DoF)
                denorm_output_motions = remake_root_position_from_displacement(
                    args, denorm_output_motions, num_bs, num_frame, num_DoF)

            # """ 4. fk loss """
            # if args.fk_loss == 1:
            #     fk = ForwardKinematics(args, file.edges)
            #     gt_transform = fk.forward_from_raw(denorm_gt_motions.permute(0,2,1), train_dataset.offsets[1][character_idx]).reshape(num_bs, -1, num_frame)
            #     output_transform = fk.forward_from_raw(denorm_output_motions.permute(0,2,1), train_dataset.offsets[1][character_idx]).reshape(num_bs, -1, num_frame)

            #     num_Transform_DoF = gt_transform.size(1)
            #     for m in range(num_bs):
            #         for j in range(num_Transform_DoF): #check dimension
            #             loss = rec_criterion(gt_transform[m][j], output_transform[m][j])
            #             loss_sum += loss
            #             fk_losses.append(loss.item())

            """ Optimization and show info """
            # check output error
            # for m in range(num_bs):
            #     for j in range(num_frame):
            #         loss = rec_criterion(
            #             gt_motions[m][j], output_motions[m][j])
            #         # loss_sum += loss
            #         # losses.append(loss.item())
            #         rec_losses.append(loss.item())

            pbar.update(1)
            pbar.set_postfix_str(
                f"mean: {np.mean(rec_losses):.3f}, G_loss: {np.mean(G_losses):.3f}, D_loss: {np.mean(D_losses):.3f}")

            """ BVH Writing """
            if epoch == 0:
                write_bvh(save_dir, "gt", denorm_gt_motions,
                          characters, character_idx, motion_idx, args)

            if epoch % 10 == 0:
                write_bvh(save_dir, "output_"+str(epoch), denorm_output_motions,
                          characters, character_idx, motion_idx, args)

        torch.cuda.empty_cache()
        del gt_motions, enc_inputs, dec_inputs, output_motions

    return np.mean(rec_losses), np.mean(G_losses), np.mean(D_losses)
