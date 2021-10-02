import json
import torch
import os
import numpy as np
from wandb import set_trace
import option_parser
from tqdm import tqdm

from datasets.bvh_parser import BVH_file
from datasets.bvh_writer import BVH_writer

from models.Kinematics import ForwardKinematics

def train_epoch(args, epoch, model, criterion, optimizer, train_loader, train_dataset, Files, bvh_writers, characters, save_name):
    losses = [] # losses for 1 epoch 
    # losses_quat = []
    elements_losses = []
    fk_losses = []
    # losses_norm = []
    # norm = []
    model.train()
    args.epoch = epoch
    character_idx = 0

    with tqdm(total=len(train_loader)-1, desc=f"TrainEpoch {epoch}") as pbar:
        save_dir = args.save_dir + save_name
        try_mkdir(save_dir)
        for i, value in enumerate(train_loader):
            optimizer.zero_grad()

            """ 1. Get Data and Set value to model and Get output """
            input_motions, gt_motions = map(lambda v : v.to(args.cuda_device), value)
            enc_inputs, dec_inputs = input_motions, input_motions
            output_motions = model(enc_inputs, dec_inputs)
            
            """ 1'. remove offset part """
            num_joint = int(output_motions.size(1)/2)
            output_motions = output_motions[:, :num_joint,:]
            gt_motions = gt_motions[:, :num_joint, :]
            
            """ 2. Get numbers """
            num_bs = gt_motions.size(0)
            num_DoF = gt_motions.size(1)
            num_frame = gt_motions.size(2)

            # Get Character Index
            motion_idx = i * args.batch_size
            character_idx = int(motion_idx / args.num_motions)

            """ 3. remake root position from displacement and denorm for bvh_writing """
            """ denorm for fk """
            if args.normalization == 1:
                denorm_gt_motions = train_dataset.denorm(1, character_idx, gt_motions)
                denorm_output_motions = train_dataset.denorm(1, character_idx, output_motions)
            else:
                denorm_gt_motions = gt_motions
                denorm_output_motions = output_motions

            """ remake root position """
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


            """ 4. Denorm and do FK and set all data """
            """ set dataset """
            # gt_all = torch.cat((gt_motions, gt_transform), dim=1)
            # output_all = torch.cat((output_motions, output_transform), dim=1)
            """ update DoF """
            # num_total_DoF = gt_all.size(1)
                       
            """ 5. Get loss (orienation & FK & regularization) """
            # transpose for quaternion loss
            # gt = denorm_gt_motions.transpose(1,2)
            # output = denorm_output_motions.transpose(1,2) # (128,91)
            loss_sum = 0

            """ 5-1. loss on each element """
            for m in range(num_bs):
                for j in range(num_DoF):
                    loss = criterion(gt_motions[m][j], output_motions[m][j])
                    loss_sum += loss
                    losses.append(loss.item())
                    elements_losses.append(loss.item())

            # """ 5-2. position loss on root """
            # gt_pos = gt[:, :, :3]
            # output_pos = output[:, :, :3]
            # loss = torch.square(gt_pos - output_pos)

            # loss = loss.reshape(-1)
            # for i in range(loss.size(0)):
            #     loss_sum += loss[i]
            #     losses.append(loss[i].item())

            """ 5-3. quatnion loss on rotation  """
            # GT: 1.rot 부분만 detach -> 2. quat 으로 만들어주기 -> 3. inverse
            # if args.rotation == 'Quaternion':
            #     gt_rot = gt[:, :, :-3]
            #     gt_rot_hat = gt_rot.reshape(gt.size(0), gt.size(1), 4, -1)
            #     gt_rot_hat[:, :, 1:4, :] = -1 * gt_rot_hat[:, :, 1:4, :]
            #     square_sum = torch.sum(torch.square(gt_rot_hat), dim=2).unsqueeze(dim=2)
            #     gt_rot_hat = (1 / square_sum) * gt_rot_hat

            #     # output motion 
            #     output_rot = output[:, :, :-3]
            #     output_rot = output_rot.reshape(gt.size(0), gt.size(1), 4, -1)

            #     # element-wise multiple
            #     q_output = (gt_rot_hat * output_rot).sum(axis=2)
            #     ones = torch.ones(gt.size(0), gt.size(1), output_rot.size(-1)).to(args.cuda_device)
            #     for m in range(num_bs):
            #         for j in range(num_frame):
            #             loss = criterion(q_output[m][j], ones[m][j])
            #             loss_sum += loss
            #             losses.append(loss.item())
            #             losses_quat.append(loss.item())
        
            """ 5-4. Regularization Loss Ter가 """
            # norm = torch.as_tensor(0.).cuda()
            # for param in model.parameters():
            #     norm += torch.norm(param)
            #     losses_norm.append(norm)
            #     loss_sum += norm

            """ 5-5. FK loss """
            """ Do FK"""
            fk = ForwardKinematics(args, Files[1][0].edges)
            gt_transform  = fk.forward_from_raw(denorm_gt_motions, train_dataset.offsets[1][0]).reshape(num_bs, -1, num_frame)
            output_transform = fk.forward_from_raw(denorm_output_motions, train_dataset.offsets[1][0]).reshape(num_bs, -1, num_frame)

            for m in range(num_bs):
                for j in range (num_DoF):
                    loss = criterion(gt_transform[m][j], output_transform[m][j])
                    loss_sum += loss
                    losses.append(loss.item())
                    fk_losses.append(loss.item())

            """ 5-6. smoothing loss """
            # for m in range(num_bs):
            #     for j in range(num_DoF):
            #         loss 

            """ 6. optimization and show info """
            loss_sum.backward()
            optimizer.step()
            pbar.update(1)
            pbar.set_postfix_str(f"elements_loss: {np.mean(elements_losses):.3f}, fk_loss: {np.mean(fk_losses):.3f} (mean: {np.mean(losses):.3f})")
            # pbar.set_postfix_str(f"elements_loss: {np.mean(losses_elements):.3f}, quat_loss: {np.mean(losses_quat):.3f} (mean: {np.mean(losses):.3f})")


            """ 7. BVH Writing : writing 형식 (bs, DoF, window)"""
            # Write gt motion for 0 epoch. 
            if epoch == 0:
                save_dir_gt = save_dir + "character_{}/gt/".format(character_idx)
                try_mkdir(save_dir_gt)
                file = BVH_file(option_parser.get_std_bvh(dataset = characters[0][character_idx]))
                bvh_writer = BVH_writer(file.edges, file.names)
                for j in range(args.batch_size):
                    file_name = save_dir_gt + "motion_{}.bvh".format(int(motion_idx % args.num_motions + j))
                    motion = denorm_gt_motions[j]
                    bvh_writer.write_raw(motion, args.rotation, file_name)

            # Write output motions for 10 epoch
            if epoch % 10 == 0:
                save_dir_output = save_dir + "character_{}/output_{}/".format(character_idx, epoch)
                try_mkdir(save_dir_output)
                file = BVH_file(option_parser.get_std_bvh(dataset = characters[0][character_idx]))
                bvh_writer = BVH_writer(file.edges, file.names)
                for j in range(args.batch_size):
                    file_name = save_dir_output + "motion_{}.bvh".format(int(motion_idx % args.num_motions + j))
                    motion = denorm_output_motions[j]
                    bvh_writer.write_raw(motion, args.rotation, file_name)
            
            torch.cuda.empty_cache()
            del gt_motions, enc_inputs, dec_inputs
    
    return np.mean(elements_losses), np.mean(fk_losses), np.mean(losses)

""" eval """
def eval_epoch(args, model, data_loader):
    matchs = []
    batch_size = data_loader.batch_size
    model.eval()

    # save_dir = "./output/eval_result/"
    # try_mkdir(save_dir)
    # save output of eval 

    # validation set: included on training dataset, test set: not on training dataset
    with tqdm(total=len(data_loader), desc=f"TestSet") as pbar: 
        for i, value in enumerate(data_loader):
            input_motion, gt_motions = map(lambda v : v.to(args.cuda_device), value)
            enc_inputs, dec_inputs = input_motion, input_motion
            outputs = model(enc_inputs, dec_inputs)

            logits = outputs[0]

            pbar.update(1)
            
            torch.cuda.empty_cache()
            del enc_inputs, dec_inputs

    # return np.sum(matchs) / len(matchs) if 0 < len(matchs) else 0

def save(model, path, epoch):
    try_mkdir(path)
    path = os.path.join(path, str(epoch))
    torch.save(model.state_dict(), path)
    # print('Save at {} succeed!'.format(path))

def load(model, path, epoch):
    path = os.path.join(path, str(epoch))
    print('loading from ', path)
    if not os.path.exists(path):
        raise Exception('Unknown loading path')
    model.load_state_dict(torch.load(path))
    print('load succeed')

def try_mkdir(path):
    if not os.path.exists(path):
        # print('make new dir')
        os.system('mkdir -p {}'.format(path))
