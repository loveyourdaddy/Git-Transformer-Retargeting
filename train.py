import torch
import os
import numpy as np
# from wandb import set_trace
from datasets import get_character_names
import option_parser
from tqdm import tqdm
from datasets.bvh_parser import BVH_file
from datasets.bvh_writer import BVH_writer
from models.Kinematics import ForwardKinematics
# from rendering import *
import torchvision
from models.utils import GAN_loss
import wandb

SAVE_ATTENTION_DIR = "attention_vis"
SAVE_ATTENTION_DIR_INTRA = "attention_vis_intra"
os.makedirs(SAVE_ATTENTION_DIR, exist_ok=True)
os.makedirs(SAVE_ATTENTION_DIR_INTRA, exist_ok=True)

# def get_data_numbers(motion):
#     return motion.size(0), motion.size(1), motion.size(2)

def get_curr_motion(iter, batch_size):
    return iter * batch_size

def get_curr_character(motion_idx, num_motions):
    return int(motion_idx / num_motions)

def denormalize(dataset, character_idx, motions, i):
    return dataset.denorm(i, character_idx, motions)

def remake_root_position_from_displacement(args, motions, num_bs, num_frame, num_DoF):
    for bs in range(num_bs):  # dim 0
        for frame in range(num_frame - 1):  # dim 2 # frame: 0~62. update 1 ~ 63
            motions[bs][frame + 1][num_DoF - 3] += motions[bs][frame][num_DoF - 3]
            motions[bs][frame + 1][num_DoF - 2] += motions[bs][frame][num_DoF - 2]
            motions[bs][frame + 1][num_DoF - 1] += motions[bs][frame][num_DoF - 1]

    return motions

def write_bvh(save_dir, gt_or_output_epoch, motion, characters, character_idx, motion_idx, args,i):
    save_dir_gt = save_dir + "character{}_{}/{}/".format(
        character_idx, characters[i][character_idx], gt_or_output_epoch)
    try_mkdir(save_dir_gt)
    file = BVH_file(option_parser.get_std_bvh(dataset=characters[i][character_idx]))

    bvh_writer = BVH_writer(file.edges, file.names)
    for j in range(args.batch_size):
        file_name = save_dir_gt + \
            "motion_{}.bvh".format(int(motion_idx % args.num_motions + j))
        bvh_writer.write_raw(motion[j], args.rotation, file_name)

def try_mkdir(path):
    if not os.path.exists(path):
        # print('make new dir')
        os.system('mkdir -p {}'.format(path))

def requires_grad_(model, requires_grad):
    for para in model.parameters():
        para.requires_grad = requires_grad
            
def train_epoch(args, epoch, modelGs, modelDs, optimizerGs, optimizerDs, train_loader, train_dataset, characters, save_name, Files):

    # Set return list 
    n_topology = len(modelGs)
    input_motions           = [0] * n_topology
    output_motions          = [0] * n_topology
    gt_motions              = [0] * n_topology
    denorm_gt_motions       = [0] * n_topology
    denorm_output_motions   = [0] * n_topology

    fk_losses     = []
    rec_losses0   = []
    rec_losses1   = []
    G_losses      = []
    D_losses_fake = []
    D_losses_real = []
    
    # set loss list 
    # fk_losses     = [0] * n_topology
    # rec_losses    = [0] * n_topology
    # G_losses      = [0] * n_topology
    # D_losses_fake = [0] * n_topology
    # D_losses_real = [0] * n_topology

    for i in range(n_topology):
        modelGs[i].train()
        modelDs[i].train()

    args.epoch = epoch
    character_idx = 0
    rec_criterion = torch.nn.MSELoss()
    gan_criterion = GAN_loss(args.gan_mode).to(args.cuda_device)

    with tqdm(total=len(train_loader), desc=f"TrainEpoch {epoch}") as pbar:
        save_dir = args.save_dir + save_name
        try_mkdir(save_dir)

        for i, value in enumerate(train_loader):

            """ Get Data and Set value to model and Get output """
            source_motions, target_motions = map(lambda v: v.to(args.cuda_device), value)

            input_motions[0] = source_motions
            input_motions[1] = target_motions
            gt_motions[0] = source_motions
            gt_motions[1] = target_motions

            # """ Get Data numbers: (bs, DoF, window) """
            num_bs, Dim1, Dim2 = target_motions.size(0), target_motions.size(1), target_motions.size(2)
            if args.swap_dim == 0:
                num_frame, num_DoF = Dim1, Dim2
            else:
                num_DoF, num_frame = Dim1, Dim2

            motion_idx = get_curr_motion(i, args.batch_size)
            character_idx = get_curr_character(motion_idx, args.num_motions)

            # for j in range(args.n_topology):
            #     optimizerGs[j].zero_grad()
            #     optimizerDs[j].zero_grad()

                # output_motions[j], enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs = modelGs[j](
                #     character_idx, character_idx, input_motions[j])

            # """ save attention map """
            # if epoch % 100 == 0:
            #     bs = enc_self_attn_probs[0].size(0)
            #     img_size = enc_self_attn_probs[0].size(2)
            #     for att_layer_index, enc_self_attn_prob in enumerate(enc_self_attn_probs):
            #         att_map = enc_self_attn_prob.view(bs*4, -1, img_size, img_size)
            #         torchvision.utils.save_image(att_map, \
            #             f"./{SAVE_ATTENTION_DIR}/enc_{att_layer_index}_{epoch:04d}.jpg", range=(0, 1), normalize=True)

            #     img_size = dec_self_attn_probs[0].size(2)
            #     for att_layer_index, dec_self_attn_prob in enumerate(dec_self_attn_probs):
            #         att_map = dec_self_attn_prob.view(bs*4, -1, img_size, img_size)
            #         torchvision.utils.save_image(att_map, \
            #             f"./{SAVE_ATTENTION_DIR}/dec_{att_layer_index}_{epoch:04d}.jpg", range=(0, 1), normalize=True)

            #     img_size = dec_enc_attn_probs[0].size(2)
            #     for att_layer_index, dec_enc_attn_prob in enumerate(dec_enc_attn_probs):
            #         att_map = dec_enc_attn_prob.view(bs*4, -1, img_size, img_size)
            #         torchvision.utils.save_image(att_map, \
            #             f"./{SAVE_ATTENTION_DIR}/enc_dec_{att_layer_index}_{epoch:04d}.jpg", range=(0, 1), normalize=True)

            """ Get LOSS (orienation & FK & regularization) """
            for j in range(args.n_topology):

                """ feed to NETWORK """
                output_motions[j], enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs = modelGs[j](
                    character_idx, character_idx, input_motions[j])

                """ loss1. loss on each element """
                if args.rec_loss == 1:                    
                    # get rec loss 
                    rec_loss = rec_criterion(gt_motions[j], output_motions[j])
                    if j == 0:
                        rec_losses0.append(rec_loss.item())
                    else:
                        rec_losses1.append(rec_loss.item()) 


                """ loss 1-2. fk loss """
                if args.fk_loss == 1:
                    # Data post-processing
                    """ 1) denorm """
                    if args.normalization == 1:
                        denorm_gt_motions[j]     = denormalize(train_dataset, character_idx, gt_motions[j], j)
                        denorm_output_motions[j] = denormalize(train_dataset, character_idx, output_motions[j], j)
                    else:
                        denorm_gt_motions[j]     = gt_motions[j]
                        denorm_output_motions[j] = output_motions[j]
                    """ 2) swap dim """
                    if args.swap_dim == 1:
                        denorm_gt_motions[j]     = torch.transpose(denorm_gt_motions[j], 1, 2)
                        denorm_output_motions[j] = torch.transpose(denorm_output_motions[j], 1, 2)

                    # Get fk 
                    file = Files[j][character_idx] 
                    fk = ForwardKinematics(args, file.edges)
                    
                    # Get transform (local)
                    gt_transform     = fk.forward_from_raw(denorm_gt_motions[j].permute(0,2,1),     train_dataset.offsets[j][character_idx]).reshape(num_bs, -1, num_frame)
                    output_transform = fk.forward_from_raw(denorm_output_motions[j].permute(0,2,1), train_dataset.offsets[j][character_idx]).reshape(num_bs, -1, num_frame)

                    # Get global pos
                    gt_global_pos = fk.from_local_to_world(gt_transform).permute(0,2,1)
                    output_global_pos = fk.from_local_to_world(output_transform).permute(0,2,1)

                    # for idx_batch in range(num_bs):
                    fk_loss = rec_criterion(gt_global_pos, output_global_pos)
                    fk_losses.append(fk_loss.item())
                        

                """ loss2. GAN Loss : (fake output: 0), (real_data: 1)  """
                if args.gan_loss == 1:                    
                    """ Generator """          
                    fake_output = modelDs[j](character_idx, character_idx, output_motions[j]) # .detach()
                    for idx_batch in range(num_bs):
                        G_loss = gan_criterion(fake_output[idx_batch], True)
                        # sum_G_loss += G_loss
                        G_losses.append(G_loss.item())

                    """ Discriminator """
                    # real 
                    real_output = modelDs[j](character_idx, character_idx, gt_motions[j])
                    for idx_batch in range(num_bs):
                        D_loss_real = gan_criterion(real_output[idx_batch], True)
                        # sum_D_loss += D_loss_real
                        D_losses_real.append(D_loss_real.item())

                    # fake
                    fake_output = modelDs[j](character_idx, character_idx, output_motions[j].detach())
                    for idx_batch in range(num_bs):
                        D_loss_fake = gan_criterion(fake_output[idx_batch], False)
                        # sum_D_loss += D_loss_fake
                        D_losses_fake.append(D_loss_fake.item())


                """ backward and optimize """
                requires_grad_(modelDs[j], False)
                generator_loss = rec_loss + fk_loss + G_loss
                optimizerGs[j].zero_grad()
                generator_loss.backward()
                optimizerGs[j].step()

                requires_grad_(modelDs[j], True)
                discriminator_loss = D_loss_real + D_loss_fake
                optimizerDs[j].zero_grad()
                discriminator_loss.backward() 
                optimizerDs[j].step()

                
            """  remake root & BVH Writing """ 
            """ 3) remake root position from displacement """
            for j in range(args.n_topology):
                if args.root_pos_disp == 1:
                    denorm_gt_motions[j] = remake_root_position_from_displacement(
                        args, denorm_gt_motions[j], num_bs, num_frame, num_DoF)
                    denorm_output_motions[j] = remake_root_position_from_displacement(
                        args, denorm_output_motions[j], num_bs, num_frame, num_DoF)
                
                """ BVH Writing """ 
                if epoch == 0:
                    write_bvh(save_dir, "gt"+str(j), denorm_gt_motions[j],
                            characters, character_idx, motion_idx, args, j)
                if epoch % 100 == 0:
                    write_bvh(save_dir, "output"+str(j)+"_"+str(epoch), denorm_output_motions[j],
                            characters, character_idx, motion_idx, args, j)

            """ show info """
            pbar.update(1)
            pbar.set_postfix_str(f"mean1: {np.mean(rec_losses0):.3f}, mean2: {np.mean(rec_losses1):.3f}, fk_loss: {np.mean(fk_losses):.3f}, G_loss: {np.mean(G_losses):.3f}, D_loss_real: {np.mean(D_losses_real):.3f}, D_loss_fake: {np.mean(D_losses_fake):.3f}")

        torch.cuda.empty_cache()
        del source_motions, gt_motions, output_motions

    return np.mean(rec_losses0), np.mean(rec_losses1), np.mean(fk_losses), np.mean(G_losses), np.mean(D_losses_real), np.mean(D_losses_fake)
