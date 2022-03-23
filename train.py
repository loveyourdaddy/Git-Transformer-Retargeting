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

SAVE_ATTENTION_DIR = "attention_vis"
SAVE_ATTENTION_DIR_INTRA = "attention_vis_intra"
os.makedirs(SAVE_ATTENTION_DIR, exist_ok=True)
os.makedirs(SAVE_ATTENTION_DIR_INTRA, exist_ok=True)

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
    if i == 0: 
        group = 'topology0/'
    else: 
        group = 'topology1/'

    save_dir = save_dir + group + "character{}_{}/{}/".format(character_idx, characters[i][character_idx], gt_or_output_epoch)
    try_mkdir(save_dir)
    file = BVH_file(option_parser.get_std_bvh(dataset=characters[i][character_idx]))

    bvh_writer = BVH_writer(file.edges, file.names)
    for j in range(args.batch_size):
        file_name = save_dir + \
            "motion_{}.bvh".format(int(motion_idx % args.num_motions + j))
        bvh_writer.write_raw(motion[j], args.rotation, file_name)

def try_mkdir(path):
    if not os.path.exists(path):
        # print('make new dir')
        os.system('mkdir -p {}'.format(path))

def requires_grad_(model, requires_grad):
    for para in model.parameters():
        para.requires_grad = requires_grad
            
def train_epoch(args, epoch, modelGs, optimizerGs, train_loader, train_dataset, characters, save_name, Files):

    # Set return list 
    n_topology = len(modelGs)

    input_motions          = [0] * n_topology
    output_motions         = [0] * n_topology
    gt_motions             = [0] * n_topology
    
    bp_input_motions       = [0] * n_topology
    bp_output_motions      = [0] * n_topology
    bp_gt_motions          = [0] * n_topology

    denorm_gt_motions      = [0] * n_topology
    denorm_output_motions  = [0] * n_topology
    denorm_gt_motions_     = [0] * n_topology
    denorm_output_motions_ = [0] * n_topology

    latent_feature         = [0] * n_topology

    # loss to backward (assign value)
    rec_loss = [0] * n_topology
    # ltc_loss = [0] * n_topology

    # set loss list to mean (append)
    rec_losses0 = [] 
    rec_losses1 = [] 
    # ltc_losses  = [] 
    # cyc_losses  = [] 

    for i in range(n_topology):
        modelGs[i].train()

    args.epoch = epoch
    character_idx = 0
    rec_criterion = torch.nn.MSELoss() 
    # ltc_criterion = torch.nn.MSELoss()
    # gan_criterion = GAN_loss(args.gan_mode).to(args.cuda_device)
    
    save_dir = args.save_dir + save_name
    try_mkdir(save_dir)

    with tqdm(total=len(train_loader), desc=f"TrainEpoch {epoch}") as pbar:
        for i, value in enumerate(train_loader):

            motion_idx = get_curr_motion(i, args.batch_size)
            character_idx = get_curr_character(motion_idx, args.num_motions) # 0~3 ? 

            """ Get Data and Set value to model and Get output """
            # ToDo: current, motion of source characters and target characters
            bp_source_motions, bp_target_motions = map(lambda v: v.to(args.cuda_device), value) 

            """ Divide motion to source and gt and input """
            for j in range(args.n_topology):
                if j == 0:
                    bp_input_motions[j] = bp_source_motions
                    bp_gt_motions[j]    = bp_source_motions
                else:
                    bp_input_motions[j] = bp_target_motions
                    bp_gt_motions[j]    = bp_target_motions

            # """ Get Data numbers: (bs, DoF, window) """
            # if j == 0:
            #     num_bs, Dim1, Dim2 = source_motions.size(0), source_motions.size(1), source_motions.size(2)
            # else:
            #     num_bs, Dim1, Dim2 = target_motions.size(0), target_motions.size(1), target_motions.size(2)

            # if args.swap_dim == 0:
            #     num_frame, num_DoF = Dim1, Dim2
            # else:
            #     num_DoF, num_frame = Dim1, Dim2
 
            for j in range(args.n_topology):
                bp_output_motions[j] = torch.zeros(bp_gt_motions[j].size()).to(args.cuda_device) # body part output 
                output_motions[j] = torch.zeros(bp_gt_motions[j].size(0), bp_gt_motions[j].size(2), bp_gt_motions[j].size(3)).to(args.cuda_device) # combined motion 
                gt_motions[j] = torch.zeros(bp_gt_motions[j].size(0), bp_gt_motions[j].size(2), bp_gt_motions[j].size(3)).to(args.cuda_device) # combined motion 

            """ Get LOSS (orienation & FK & regularization) """
            """ feed to NETWORK """
            for j in range(args.n_topology):
                for b in range(6):
                    bp_output_motions[j][:,b,:,:], _ = modelGs[j].body_part_generator[b](character_idx, character_idx, bp_input_motions[j][:, b, :, :])
          
            """ loss1. loss on each element """ 
            if args.rec_loss == 1:
                # get rec loss
                for j in range(args.n_topology):
                    loss = rec_criterion(bp_gt_motions[j], bp_output_motions[j])
                    rec_loss[j] = loss
                    if j == 0:
                        rec_losses0.append(loss.item())
                    else:
                        rec_losses1.append(loss.item()) 

            """ backward and optimize """
            for j in range(args.n_topology):
                if j == 0:
                    generator_loss = (rec_loss[j])
                    optimizerGs[j].zero_grad()
                    generator_loss.backward()
                    optimizerGs[j].step()
                else:
                    generator_loss = (rec_loss[j])
                    optimizerGs[j].zero_grad()
                    generator_loss.backward()
                    optimizerGs[j].step()

            """ Combine part by motion back """  
            """ gt motion """
            index = train_dataset.groups_body_parts_index # index of joint for each body part
            length = bp_input_motions[j].size(2) # 91 
            for j in range(args.n_topology):
                # root rotation
                for k in range(4): # root rotation 0~3
                    gt_motions[j][:, k, :] = bp_gt_motions[j][:, 0, k, :]
                # root position
                for k in range(3): # position len-3 ~ -1
                    gt_motions[j][:, length - 3 + k, :] = bp_gt_motions[j][:, 0, length - 3 + k, :]
                # body parts
                for b in range(5):
                    gt_motions[j][:, index[1][character_idx][b], :] = bp_gt_motions[j][:, b+1, index[1][character_idx][b], :]

            """ output motion """
            for j in range(args.n_topology):
                # root rotation
                for k in range(4): # root rotation 0~3
                    output_motions[j][:, k, :] = bp_output_motions[j][:, 0, k, :]
                # root position
                for k in range(3): # position len-3 ~ -1
                    output_motions[j][:, length - 3 + k, :] = bp_output_motions[j][:, 0, length - 3 + k, :]
                # body parts
                for b in range(5):
                    output_motions[j][:, index[1][character_idx][b], :] = bp_output_motions[j][:, b+1, index[1][character_idx][b], :]
            
            """ Remake root & BVH Writing """
            if epoch % 50 == 0:
                for j in range(args.n_topology):
                    """ 1) denorm """
                    if args.normalization == 1:
                        denorm_gt_motions[j]     = denormalize(train_dataset, character_idx, gt_motions[j], j)
                        denorm_output_motions[j] = denormalize(train_dataset, character_idx, output_motions[j], j)
                    else:
                        denorm_gt_motions[j]     = gt_motions[j]
                        denorm_output_motions[j] = output_motions[j]
                    
                    """ 2) swap dim """
                    if args.swap_dim == 1:
                        denorm_gt_motions_[j]     = torch.transpose(denorm_gt_motions[j], 1, 2)
                        denorm_output_motions_[j] = torch.transpose(denorm_output_motions[j], 1, 2)
                    else:
                        denorm_gt_motions_[j]     = denorm_gt_motions[j]
                        denorm_output_motions_[j] = denorm_output_motions[j]
                    
                    """ 3) remake root position from displacement """
                    # if args.root_pos_disp == 1:
                    #     denorm_gt_motions[j]     = remake_root_position_from_displacement(args, denorm_gt_motions_[j], num_bs, num_frame, num_DoF)
                    #     denorm_output_motions[j] = remake_root_position_from_displacement(args, denorm_output_motions_[j], num_bs, num_frame, num_DoF)
                    
                    """ BVH Writing """ 
                    if epoch == 0:
                        write_bvh(save_dir, "gt", denorm_gt_motions_[j], characters, character_idx, motion_idx, args, j)
                    if epoch != 0:
                        write_bvh(save_dir, "output"+str(epoch), denorm_output_motions_[j], characters, character_idx, motion_idx, args, j)

            # """Check """
            # loss = rec_criterion(gt_motions[1], output_motions[1])            
            # rec_losses1.append(loss.item())

            """ show info """
            pbar.update(1)
            pbar.set_postfix_str(
                f"mean1: {np.mean(rec_losses0):.7f}, mean2: {np.mean(rec_losses1):.7f}, {rec_loss[0]:.3f}, {rec_loss[1]:.3f}")

        torch.cuda.empty_cache()
        del bp_source_motions, bp_gt_motions, output_motions, bp_output_motions, latent_feature, denorm_gt_motions, denorm_gt_motions_ 

    return np.mean(rec_losses0), np.mean(rec_losses1)
