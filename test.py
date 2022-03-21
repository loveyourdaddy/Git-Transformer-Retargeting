import json
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
from train import *

SAVE_ATTENTION_DIR = "attention_vis/test"
os.makedirs(SAVE_ATTENTION_DIR, exist_ok=True)

""" eval """
def eval_epoch(args, epoch, modelGs, train_loader, train_dataset, characters, save_name, Files):
    
    n_topology = len(modelGs)
    # losses = []  # losses for test epoch # 매 스텝마다 초기화 되는 loss들
    input_motions           = [0] * n_topology
    output_motions          = [0] * n_topology
    gt_motions              = [0] * n_topology
    rec_loss                = [0] * n_topology
    denorm_output_motions   = [0] * n_topology
    denorm_output_motions_  = [0] * n_topology

    rec_losses0 = [] 
    rec_losses1 = [] 
    # cyc_losses  = [] 

    for i in range(n_topology):
        modelGs[i].eval()

    save_dir = args.save_dir + save_name
    try_mkdir(save_dir)
    
    rec_criterion = torch.nn.MSELoss()

    with tqdm(total=len(train_loader), desc=f"TestEpoch {epoch}") as pbar:
        for i, value in enumerate(train_loader):

            source_motions, target_motions = map(lambda v: v.to(args.cuda_device), value)

            for j in range(args.n_topology):
                if j == 0 : 
                    input_motions[j] = source_motions
                    gt_motions[j] = source_motions
                else:
                    input_motions[j] = target_motions
                    gt_motions[j] = target_motions

            # """ Get Data numbers: (bs, DoF, window) """
            if j == 0 : 
                num_bs, Dim1, Dim2 = source_motions.size(0), source_motions.size(1), source_motions.size(2)
            else:
                num_bs, Dim1, Dim2 = target_motions.size(0), target_motions.size(1), target_motions.size(2)
                
            if args.swap_dim == 0:
                num_frame, num_DoF = Dim1, Dim2
            else:
                num_DoF, num_frame = Dim1, Dim2

            motion_idx = get_curr_motion(i, args.batch_size)
            character_idx = get_curr_character(motion_idx, args.num_motions)

            # topology 1 
            output_motions[0], _ = modelGs[0](character_idx, character_idx, input_motions[0])
            output_motions[1], _ = modelGs[1](character_idx, character_idx, input_motions[1])

            # # topology1 -> topology 2
            # latent_feature, _, _ = modelGs[0].transformer.encoder(character_idx, input_motions[0], data_encoding=1)
            # output_motions[1], _, _ = modelGs[1].transformer.decoder(character_idx, latent_feature, latent_feature, data_encoding=1)

            for j in range(args.n_topology):
                loss = rec_criterion(gt_motions[j], output_motions[j])
                rec_loss[j] = loss
                if j == 0:
                    rec_losses0.append(loss.item())
                else:
                    rec_losses1.append(loss.item()) 

            # loss = rec_criterion(ltc, latent_feature)
            
            """  remake root & BVH Writing """ 
            for j in range(args.n_topology):
                """ 1) denorm """
                if args.normalization == 1:
                    denorm_output_motions[j] = denormalize(train_dataset, character_idx, output_motions[j], j)
                else:
                    denorm_output_motions[j] = output_motions[j]
                
                """ 2) swap dim """
                if args.swap_dim == 1:
                    denorm_output_motions_[j] = torch.transpose(denorm_output_motions[j], 1, 2)
                else:
                    denorm_output_motions_[j] = denorm_output_motions[j]
                
                """ 3) remake root position from displacement """
                if args.root_pos_disp == 1:
                    denorm_output_motions[j] = remake_root_position_from_displacement(
                        args, denorm_output_motions_[j], num_bs, num_frame, num_DoF)

                write_bvh(save_dir, "test", denorm_output_motions_[j],
                        characters, character_idx, motion_idx, args, j)

            pbar.update(1)
            pbar.set_postfix_str(f"mean1: {np.mean(rec_losses0):.7f}, mean2: {np.mean(rec_losses1):.7f}")
    # return np.mean(rec_losses0), np.mean(rec_losses1)
