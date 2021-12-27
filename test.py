import json
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
from train import *

""" eval """
# fk_losses_all = []
def eval_epoch(args, model, criterion, test_dataset, data_loader, characters, save_name, Files):
    model.eval()
    losses = [] # losses for test epoch
    fk_losses = []
    fk_loss_by_motions = []
    with tqdm(total=len(data_loader), desc=f"TestSet") as pbar:
        for i, value in enumerate(data_loader):
            # 매 스텝마다 초기화 되는 loss들             
            # denorm_losses_ = []
            
            input_motion, gt_motions = map(lambda v : v.to(args.cuda_device), value)
            enc_inputs, dec_inputs = input_motion, input_motion

            # """ Get Data numbers: (bs, DoF, window) """
            num_bs, num_DoF, num_frame = get_data_numbers(gt_motions)
            motion_idx = get_curr_motion(i, args.batch_size) 
            character_idx = get_curr_character(motion_idx, args.num_motions)
            file = Files[1][character_idx]

            """  feed to network"""
            input_character, output_character = character_idx, character_idx
            output_motions = model(input_character, output_character, enc_inputs, dec_inputs)

            """ Post-process data  """
            # """ remove offset part of output motions """
            output_motions = output_motions[:,:,:num_frame]

            # """ remake root position from displacement and denorm for bvh_writing """
            if args.normalization == 1:
                denorm_gt_motions = denormalize(test_dataset, character_idx, gt_motions)
                denorm_output_motions = denormalize(test_dataset, character_idx, output_motions)
            else:
                denorm_gt_motions = gt_motions
                denorm_output_motions = output_motions

            # """ remake root position from displacement """
            if args.root_pos_disp == 1:
                denorm_gt_motions = remake_root_position_from_displacement(denorm_gt_motions, num_bs, num_DoF, num_frame)
                denorm_output_motions = remake_root_position_from_displacement(denorm_output_motions, num_bs, num_DoF, num_frame)

            """ 1. Get loss (orienation & FK & regularization) """
            loss_sum = 0

            # """ 5-1. loss on each element """
            for m in range(num_bs):
                for j in range(num_DoF):
                    loss = criterion(gt_motions[m][j], output_motions[m][j])
                    loss_sum += loss
                    losses.append(loss.item())

            """ 2. fk loss """
            if args.fk_loss == 1:
                fk = ForwardKinematics(args, file.edges)
                gt_transform = fk.forward_from_raw(denorm_gt_motions.permute(0,2,1), test_dataset.offsets[1][character_idx]).reshape(num_bs, -1, num_frame)
                output_transform = fk.forward_from_raw(denorm_output_motions.permute(0,2,1), test_dataset.offsets[1][character_idx]).reshape(num_bs, -1, num_frame)

                num_DoF = gt_transform.size(1)
                for m in range(num_bs):
                    for j in range(num_DoF): #check dimension 
                        loss = criterion(gt_transform[m][j], output_transform[m][j])
                        loss_sum += loss
                        fk_losses.append(loss.item())

                """ Rendering FK result """
                # 16,69,128 -> 16,128,69
                gt_transform = gt_transform.permute(0,2,1)
                output_transform = output_transform.permute(0,2,1)

                if args.render == True:
                    # render 1 frame 
                    render_dots(gt_transform[0][0].reshape(-1,3)) # divide 69 -> 23,3

            # permute 
            # denorm_gt_motions = denorm_gt_motions.permute(0,2,1)
            # denorm_output_motions = denorm_output_motions.permute(0,2,1)

            num_DoF = gt_transform.size(1)
            for m in range(num_bs):
                # fk_loss = []
                for j in range(num_DoF):
                    loss = criterion(gt_transform[m][j], output_transform[m][j])
                    fk_losses.append(loss.item())
                fk_loss_by_motions.append(np.mean(fk_losses))
 
            """ show info """
            pbar.update(1)
            pbar.set_postfix_str(f"denorm_loss: {np.mean(fk_losses):.3f}, (mean: {np.mean(losses):.3f})")
            
            """ BVH Writing """
            save_dir = args.save_dir + save_name
            write_bvh(save_dir, "test_gt", denorm_gt_motions, characters, character_idx, motion_idx, args)
            write_bvh(save_dir, "test_output", denorm_output_motions, characters, character_idx, motion_idx, args)

            # del 
        torch.cuda.empty_cache()
        del enc_inputs, dec_inputs

    print("retargeting loss: {}".format(np.mean(fk_losses)))
    # return np.sum(matchs) / len(matchs) if 0 < len(matchs) else 0

def try_mkdir(path):
    if not os.path.exists(path):
        # print('make new dir')
        os.system('mkdir -p {}'.format(path))