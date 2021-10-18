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

def get_data_numbers(motion):
    return motion.size(0), motion.size(1), motion.size(2)

def get_curr_motion(iter, batch_size):
    return iter * batch_size

def get_curr_character(motion_idx, num_motions):
    return int(motion_idx / num_motions)

def denormalize(dataset, character_idx, motions):
    return dataset.denorm(1, character_idx, motions)

def remake_root_position_from_displacement(motions, num_bs, num_DoF, num_frame):

    for bs in range(num_bs): # dim 0
        for frame in range(num_frame - 1): # dim 2 # frame: 0~62. update 1 ~ 63
            motions[bs][num_DoF - 3][frame + 1] += motions[bs][num_DoF - 3][frame]
            motions[bs][num_DoF - 2][frame + 1] += motions[bs][num_DoF - 2][frame]
            motions[bs][num_DoF - 1][frame + 1] += motions[bs][num_DoF - 1][frame]

    return motions

def write_bvh(save_dir, gt_or_output_epoch, motion, characters, character_idx, motion_idx, args):
    save_dir_gt = save_dir + "character{}_{}/{}/".format(character_idx, characters[1][character_idx], gt_or_output_epoch)
    try_mkdir(save_dir_gt)
    file = BVH_file(option_parser.get_std_bvh(dataset = characters[1][character_idx]))
    bvh_writer = BVH_writer(file.edges, file.names)
    for j in range(args.batch_size):
        file_name = save_dir_gt + "motion_{}.bvh".format(int(motion_idx % args.num_motions + j))
        bvh_writer.write_raw(motion[j], args.rotation, file_name)
    
def train_epoch(args, epoch, model, criterion, optimizer, train_loader, train_dataset, characters, save_name):
    losses = [] # losses for 1 epoch
    elements_losses = []
    fk_losses = []
    model.train()
    args.epoch = epoch
    character_idx = 0

    with tqdm(total=len(train_loader), desc=f"TrainEpoch {epoch}") as pbar:
        save_dir = args.save_dir + save_name
        try_mkdir(save_dir)
        for i, value in enumerate(train_loader):
            optimizer.zero_grad()

            """ Get Data and Set value to model and Get output """
            input_motions, gt_motions = map(lambda v : v.to(args.cuda_device), value)
            enc_inputs, dec_inputs = input_motions, input_motions

            # """ Get Data numbers: (bs, DoF, window) """
            num_bs, num_DoF, num_frame = get_data_numbers(gt_motions)
            motion_idx = get_curr_motion(i, args.batch_size) 
            character_idx = get_curr_character(motion_idx, args.num_motions)

            """  feed to network"""
            input_character, output_character = character_idx, character_idx
            output_motions = model(input_character, output_character, enc_inputs, dec_inputs)

            """ Post-process data  """
            # """ remove offset part of output motions """
            output_motions = output_motions[:,:,:num_frame]

            # """ remake root position from displacement and denorm for bvh_writing """
            if args.normalization == 1:
                denorm_gt_motions = denormalize(train_dataset, character_idx, gt_motions)
                denorm_output_motions = denormalize(train_dataset, character_idx, output_motions)
            else:
                denorm_gt_motions = gt_motions
                denorm_output_motions = output_motions

            # """ remake root position from displacement """
            if args.root_pos_disp == 1:
                denorm_gt_motions = remake_root_position_from_displacement(denorm_gt_motions, num_bs, num_DoF, num_frame)
                denorm_output_motions = remake_root_position_from_displacement(denorm_output_motions, num_bs, num_DoF, num_frame)

            """ Get loss (orienation & FK & regularization) """
            loss_sum = 0

            # """ 5-1. loss on each element """
            for m in range(num_bs):
                for j in range(num_DoF):
                    loss = criterion(gt_motions[m][j], output_motions[m][j])
                    loss_sum += loss
                    losses.append(loss.item())
                    elements_losses.append(loss.item())

            # """ 5-2.same displacement loss """ 
            # same displacement between gt / output 
            # or, displacement is similar with previous loss 

            """ Optimization and show info """
            loss_sum.backward()
            optimizer.step()
            pbar.update(1)
            pbar.set_postfix_str(f"elements_loss: {np.mean(elements_losses):.3f}, fk_loss: {np.mean(fk_losses):.3f} (mean: {np.mean(losses):.3f})")

            """ BVH Writing """
            # Write gt motion for 0 epoch
            if epoch == 0:
                write_bvh(save_dir, "gt", denorm_gt_motions, characters, character_idx, motion_idx, args)

            # Write output motions for every 10 epoch
            if epoch % 10 == 0:
                write_bvh(save_dir, "output_"+str(epoch), denorm_output_motions, characters, character_idx, motion_idx, args)

        torch.cuda.empty_cache()
        del gt_motions, enc_inputs, dec_inputs, output_motions

    return np.mean(elements_losses), np.mean(fk_losses), np.mean(losses)
    # return np.mean(elements_losses), np.mean(fk_losses), np.mean(losses)

""" eval """
def eval_epoch(args, model, criterion, test_dataset, data_loader, characters, save_name):
    model.eval()
    losses = [] # losses for 1 epoch
    denorm_losses = []
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

            """ Get loss (orienation & FK & regularization) """
            loss_sum = 0

            # """ 5-1. loss on each element """
            for m in range(num_bs):
                for j in range(num_DoF):
                    loss = criterion(gt_motions[m][j], output_motions[m][j])
                    loss_sum += loss
                    losses.append(loss.item())
                    # elements_losses.append(loss.item())

            for m in range(num_bs):
                for j in range(num_DoF):
                    loss = criterion(denorm_gt_motions[m][j], denorm_output_motions[m][j])
                    denorm_losses.append(loss.item())

            pbar.update(1)
            pbar.set_postfix_str(f"denorm_loss: {np.mean(denorm_losses):.3f}, (mean: {np.mean(losses):.3f})")
            
            """ BVH Writing """
            # Write gt motion for 0 epoch
            save_dir = args.save_dir + save_name
            write_bvh(save_dir, "test_gt", denorm_gt_motions, characters, character_idx, motion_idx, args)

            # Write output motions for every 10 epoch
            write_bvh(save_dir, "test_output", denorm_output_motions, characters, character_idx, motion_idx, args)

            # del 
            torch.cuda.empty_cache()
            del enc_inputs, dec_inputs

    # return np.sum(matchs) / len(matchs) if 0 < len(matchs) else 0

def try_mkdir(path):
    if not os.path.exists(path):
        # print('make new dir')
        os.system('mkdir -p {}'.format(path))