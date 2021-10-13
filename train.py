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
        import pdb; pdb.set_trace()
        for i, value in enumerate(train_loader):
            optimizer.zero_grad()

            """ Get Data and Set value to model and Get output """
            input_motions, gt_motions = map(lambda v : v.to(args.cuda_device), value)
            enc_inputs, dec_inputs = input_motions, input_motions

            # """ Get Data numbers """
            num_bs = gt_motions.size(0)
            num_DoF = gt_motions.size(1)
            num_frame = gt_motions.size(2)

            motion_idx = i * args.batch_size
            character_idx = int(motion_idx / args.num_motions)

            """  feed to network"""
            input_character, output_character = character_idx, character_idx
            output_motions = model(input_character, output_character, enc_inputs, dec_inputs)

            """ Preocess data  """
            # """ remove offset part """
            output_motions = output_motions[:,:,:num_frame]

            # """ remake root position from displacement and denorm for bvh_writing """
            if args.normalization == 1:
                denorm_gt_motions = train_dataset.denorm(1, character_idx, gt_motions)
                denorm_output_motions = train_dataset.denorm(1, character_idx, output_motions)
            else:
                denorm_gt_motions = gt_motions
                denorm_output_motions = output_motions

            # """ remake root position """
            # data: (bs, DoF, window)
            if args.root_pos_disp == 1:
                for bs in range(num_bs): # dim 0
                    for frame in range(num_frame - 1): # dim 2
                        denorm_gt_motions[bs][num_DoF - 3][frame + 1] += denorm_gt_motions[bs][num_DoF - 3][frame]
                        denorm_gt_motions[bs][num_DoF - 2][frame + 1] += denorm_gt_motions[bs][num_DoF - 2][frame]
                        denorm_gt_motions[bs][num_DoF - 1][frame + 1] += denorm_gt_motions[bs][num_DoF - 1][frame]

                        denorm_output_motions[bs][num_DoF - 3][frame + 1] += denorm_output_motions[bs][num_DoF - 3][frame]
                        denorm_output_motions[bs][num_DoF - 2][frame + 1] += denorm_output_motions[bs][num_DoF - 2][frame]
                        denorm_output_motions[bs][num_DoF - 1][frame + 1] += denorm_output_motions[bs][num_DoF - 1][frame]

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

            """ BVH Writing : writing 형식 (bs, DoF, window)"""
            # Write gt motion for 0 epoch.
            if epoch == 0:
                save_dir_gt = save_dir + "character{}_{}/gt/".format(character_idx, characters[1][character_idx])
                try_mkdir(save_dir_gt)
                file = BVH_file(option_parser.get_std_bvh(dataset = characters[1][character_idx]))
                bvh_writer = BVH_writer(file.edges, file.names)
                for j in range(args.batch_size):
                    file_name = save_dir_gt + "motion_{}.bvh".format(int(motion_idx % args.num_motions + j))
                    motion = denorm_gt_motions[j]
                    bvh_writer.write_raw(motion, args.rotation, file_name)

            # Write output motions for 10 epoch
            if epoch % 10 == 0:
                save_dir_output = save_dir + "character{}_{}/output_{}/".format(character_idx, characters[1][character_idx], epoch)
                try_mkdir(save_dir_output)
                file = BVH_file(option_parser.get_std_bvh(dataset = characters[1][character_idx])) # get target output
                bvh_writer = BVH_writer(file.edges, file.names)
                for j in range(args.batch_size):
                    file_name = save_dir_output + "motion_{}.bvh".format(int(motion_idx % args.num_motions + j))
                    motion = denorm_output_motions[j]
                    bvh_writer.write_raw(motion, args.rotation, file_name)

        torch.cuda.empty_cache()
        del gt_motions, enc_inputs, dec_inputs, output_motions

    return np.mean(elements_losses), np.mean(fk_losses), np.mean(losses)

""" eval """
def eval_epoch(args, model, dataset, data_loader, characters, save_name):
    model.eval()

    with tqdm(total=len(data_loader)-1, desc=f"TestSet") as pbar:
        for i, value in enumerate(data_loader):
            input_motion, gt_motions = map(lambda v : v.to(args.cuda_device), value)
            enc_inputs, dec_inputs = input_motion, input_motion

            output_motions = model(enc_inputs, dec_inputs)
            pbar.update(1)

            # get data number 
            motion_idx = i * args.batch_size
            character_idx = int(motion_idx / args.num_motions)

            num_bs = output_motions.size(0)
            num_DoF = output_motions.size(1)
            num_frame = output_motions.size(2)
            
            """ 3. remake root position from displacement and denorm for bvh_writing """
            """ denorm for fk """
            if args.normalization == 1:
                denorm_output_motions = dataset.denorm(1, character_idx, output_motions)
            else:
                denorm_output_motions = output_motions

            """ remake root position """
            # data: (bs, DoF, window)
            if args.root_pos_disp == 1:
                for bs in range(num_bs): # dim 0
                    for frame in range(num_frame - 1): # dim 2
                        denorm_output_motions[bs][num_DoF - 3][frame + 1] += denorm_output_motions[bs][num_DoF - 3][frame]
                        denorm_output_motions[bs][num_DoF - 2][frame + 1] += denorm_output_motions[bs][num_DoF - 2][frame]
                        denorm_output_motions[bs][num_DoF - 1][frame + 1] += denorm_output_motions[bs][num_DoF - 1][frame]

            # write 
            save_dir = args.save_dir + save_name
            save_dir_gt = save_dir + "character{}_{}/test/".format(character_idx, characters[1][character_idx])
            try_mkdir(save_dir_gt)
            file = BVH_file(option_parser.get_std_bvh(dataset=characters[1][character_idx]))
            
            bvh_writer = BVH_writer(file.edges, file.names)
            for j in range(args.batch_size):
                file_name = save_dir_gt + "motion_{}.bvh".format(int(motion_idx % args.num_motions + j))
                motion = denorm_output_motions[j]
                bvh_writer.write_raw(motion, args.rotation, file_name)

            # del 
            torch.cuda.empty_cache()
            del enc_inputs, dec_inputs

    # return np.sum(matchs) / len(matchs) if 0 < len(matchs) else 0

def try_mkdir(path):
    if not os.path.exists(path):
        # print('make new dir')
        os.system('mkdir -p {}'.format(path))
