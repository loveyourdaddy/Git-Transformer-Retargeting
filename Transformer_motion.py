import json
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from datasets import bvh_writer
import option_parser
from datasets import get_character_names, create_dataset
from model import MotionGenerator
# from torch.utils.tensorboard import SummaryWriter
from datasets.bvh_parser import BVH_file
from datasets.bvh_writer import BVH_writer
from models.Kinematics import ForwardKinematics
import wandb

""" motion data collate function """
def motion_collate_fn(inputs): # input ???
    # inputs : (32 batch, 2 character groups, 2 (motion, skeleton idx), 913 frames , 91 rotations and positoin of joints)
    input_motions, output_motions = list(zip(*inputs)) #? output이 2개인 이유?

    # TODO: padding 없애고 텐서로만 작업하기 
    input_motions = torch.nn.utils.rnn.pad_sequence(input_motions, batch_first=True, padding_value=0)
    output_motions = torch.nn.utils.rnn.pad_sequence(output_motions, batch_first=True, padding_value=0)
    # dec_inputs = torch.nn.utils.rnn.pad_sequence(enc_inputs, batch_first=True, padding_value=0)

    batch = [
        input_motions,
        output_motions
        # dec_inputs
    ]
    return batch

def train_epoch(args, epoch, model, criterion, optimizer, train_loader, train_dataset, Files, bvh_writers, characters, save_name):
    losses = [] # losses for 1 epoch 
    # losses_ori = []
    # losses_fk = []
    norm = []
    model.train()
    # root_weight = args.root_weight
    
    with tqdm(total=len(train_loader)-1, desc=f"TrainEpoch {epoch}") as pbar:
        save_dir = args.save_dir + save_name
        try_mkdir(save_dir)
        for i, value in enumerate(train_loader):
            optimizer.zero_grad()

            """ 1. Get Data and Set value to model and Get output """
            input_motions, gt_motions = map(lambda v : v.to(args.cuda_device), value)            
            enc_inputs, dec_inputs = input_motions, input_motions
            output_motions = model(enc_inputs, dec_inputs)
            
            """ root position & Joint orientaion loss """
            # for j in range(num_bs): # For all motions
            #     for k in range(num_DoF):  # loss for all joints of 1 frame
            #         loss = criterion(output_motions[j][k], gt_motions[j][k])

            #         """ Root의 경우 weight을 추가해줍니다. """
            #         if k == num_DoF-2 or k == num_DoF-1 or k == num_DoF:
            #             # if args.weight_root_loss:
            #             #     loss *= root_weight
            #             loss_pos.append(loss.item())
                        
            #         loss_sum += loss
            #         losses.append(loss.item())
            
            """ 2. Get numbers """
            num_bs = gt_motions.size(0)
            num_DoF = gt_motions.size(1)
            num_frame = gt_motions.size(2)
            
            """ 3. Do FK """
            # fk = ForwardKinematics(args, Files[1][0].edges)
            # gt_transform  = fk.forward_from_raw(denorm_gt_motions, train_dataset.offsets[1][0]).detach().reshape(num_bs, num_frame, -1)
            # output_transform = fk.forward_from_raw(denorm_output_motions, train_dataset.offsets[1][0]).detach().reshape(num_bs, num_frame, -1)

            """ 4. Get loss (orienation & FK & regularization) """
            # gt_all = torch.cat((gt_motions, gt_transform), dim=-1)
            # output_all = torch.cat((output_motions, output_transform), dim=-1)

            loss_sum = 0
            for m in range(num_bs):
                for j in range(num_DoF):
                    loss = criterion(gt_motions[m][j], output_motions[m][j])
                    loss_sum += loss
                    losses.append(loss.item())

            # loss_ori = criterion(gt_all[:, :, :num_DoF], output_all[:, :, :num_DoF])
            # loss_fk = criterion(gt_all[:, :, num_DoF:], output_all[:, :, num_DoF:])

            """ Regularization Loss Term """
            norm = torch.as_tensor(0.).cuda()
            for param in model.parameters():
                norm += torch.norm(param)
            loss_sum += norm

            """ 5. optimization and show info"""
            loss_sum.backward()
            optimizer.step()
            pbar.update(1)
            pbar.set_postfix_str(f"Reg_loss: {norm:.3f}, (mean: {np.mean(losses):.3f})")
            # pbar.set_postfix_str(f"Loss_ori: {np.mean(losses_ori):.3f}, Reg_loss: {norm:.3f}, (mean: {np.mean(losses):.3f})")
            # pbar.set_postfix_str(f"Loss_ori: {loss_ori:.3f}, loss_fk: {loss_fk:.3f}, Reg_loss: {norm:.3f}, (mean: {np.mean(losses):.3f})")


            """ 6. remake root position from displacement and denorm for bvh_writing """
            """ denorm """
            # Get Character Index
            motion_idx = int(i * args.batch_size)
            character_idx = int(motion_idx / args.num_motions)
            if args.normalization == 1:
                gt_motions = train_dataset.denorm(1, character_idx, gt_motions)
                output_motions = train_dataset.denorm(1, character_idx, output_motions)

            """ remake root position """
            # data: (bs, DoF, window)
            if args.root_pos_disp == 1:
                for bs in range(num_bs): # dim 0
                    for frame in range(num_frame - 1): # dim 2
                        gt_motions[bs][num_DoF - 3][frame + 1] += gt_motions[bs][num_DoF - 3][frame]
                        gt_motions[bs][num_DoF - 2][frame + 1] += gt_motions[bs][num_DoF - 2][frame]
                        gt_motions[bs][num_DoF - 1][frame + 1] += gt_motions[bs][num_DoF - 1][frame]

                        output_motions[bs][num_DoF - 3][frame + 1] += output_motions[bs][num_DoF - 3][frame]
                        output_motions[bs][num_DoF - 2][frame + 1] += output_motions[bs][num_DoF - 2][frame]
                        output_motions[bs][num_DoF - 1][frame + 1] += output_motions[bs][num_DoF - 1][frame]


            """ change data dimensiton : DoF (2) -> window (1) """
            # 데이터의 1차원과 2차원을 바꿈 for bvh writing (bs, window, DoF) -> (BS, DoF, Window)
            # output_motions = output_motions.permute(0, 2, 1)
            # gt_motions = gt_motions.permute(0, 2, 1)

            """ 7. BVH Writing : writing 형식 (bs, DoF, window)"""
            # Write gt motion for 0 epoch. 
            if epoch == 0:
                save_dir_gt = save_dir + "character_{}/gt/".format(character_idx)
                try_mkdir(save_dir_gt)
                file = BVH_file(option_parser.get_std_bvh(dataset = characters[0][character_idx]))
                bvh_writer = BVH_writer(file.edges, file.names)
                for j in range(args.batch_size):
                    file_name = save_dir_gt + "motion_{}.bvh".format(int(motion_idx % args.num_motions + j))
                    motion = gt_motions[j]
                    bvh_writer.write_raw(motion, 'quaternion', file_name)

            # Write output motions for 10 epoch
            if epoch % 10 == 0:
                save_dir_output = save_dir + "character_{}/output_{}/".format(character_idx, epoch)
                try_mkdir(save_dir_output)
                file = BVH_file(option_parser.get_std_bvh(dataset = characters[0][character_idx]))
                bvh_writer = BVH_writer(file.edges, file.names)
                for j in range(args.batch_size):
                    file_name = save_dir_output + "motion_{}.bvh".format(int(motion_idx % args.num_motions + j))
                    motion = output_motions[j]
                    bvh_writer.write_raw(motion, 'quaternion', file_name)
            
            torch.cuda.empty_cache()
            del gt_motions, enc_inputs, dec_inputs
    
    return np.mean(losses)

def try_mkdir(path):
    if not os.path.exists(path):
        # print('make new dir')
        os.system('mkdir -p {}'.format(path))

""" eval """
def eval_epoch(args, epoch, model, data_loader, data_dataset, vocab):
    matchs = []
    batch_size = data_loader.batch_size
    model.eval()

    save_dir = "./output/eval_result/"
    try_mkdir(save_dir)

    # validation set: included on training dataset, test set: not on training dataset
    with tqdm(total=len(data_loader), desc=f"TestSet") as pbar: 
        for i, value in enumerate(data_loader):
            labels, enc_inputs, dec_inputs = map(lambda v : v.to(args.cuda_device), value)

            outputs = model(enc_inputs, dec_inputs)

            logits = outputs[0]

            # max value, index
            _, indices = logits.max(1)
            
            match = torch.eq(indices, labels).detach()
            matchs.extend(match.cpu())
            accuracy = np.sum(matchs) / len(matchs) if len(matchs) > 0 else 0

            pbar.update(1)
            pbar.set_postfix_str(f"Acc: {accuracy}:.3f")

            if labels.shape[0] is not batch_size:
                return accuracy

            # save the result as text 
            path = save_dir + str(i) +".txt"
            f = open(path, 'w')
            for j in range(0, batch_size):
                f.write(str(j) + '\n')
                f.write('\t input : ' + str(data_dataset.GetText(batch_size * i + j)) + '\n')
                f.write('\t label : ' + str(labels[j]) + '\n')
                f.write('\t logits: ' + str(logits[j]) + '\n')
                f.write('\t logits: ' + vocab.id_to_piece(logits[j]) + '\n')  #check it is string
            f.close()

            #del
            torch.cuda.empty_cache()
            del labels, enc_inputs, dec_inputs

    return np.sum(matchs) / len(matchs) if 0 < len(matchs) else 0

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


""" 0. Set Env Parameters """
args = option_parser.get_args()
args.cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_path = os.path.join(args.save_dir, 'logs/')
wandb.init(project='transformer-retargeting', entity='loveyourdaddy')

print("cuda availiable: {}".format(torch.cuda.is_available()))
print("device: ", args.cuda_device)

""" Changable Parameters """
args.is_train = True # False 
path = "./parameters/"
save_name = "210916_linear4_orientation_loss_before_denorm/"

""" 1. load Motion Dataset """
characters = get_character_names(args)
train_dataset = create_dataset(args, characters)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=motion_collate_fn)
print("characters:{}".format(characters))

""" 2.Set Learning Parameters  """
args.DoF = train_dataset.GetDoF()
args.n_output = args.DoF # 91
args.num_motions = len(train_dataset) / 4

""" 3. Train and Test  """
model = MotionGenerator(args, characters, train_dataset)
model.to(args.cuda_device)
wandb.watch(model)

criterion = torch.nn.MSELoss() # torch.nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)

# Set BVH writers
BVHWriters = []
Files = []
for i in range(len(characters)):
    bvh_writers = [] 
    files = []
    for j in range(len(characters[0])):
        file = BVH_file(option_parser.get_std_bvh(dataset=characters[i][j]))
        files.append(file)
        bvh_writers.append(BVH_writer(file.edges, file.names))

    Files.append(files)
    BVHWriters.append(bvh_writers)


if args.is_train is True:
    print("cuda device: ", args.cuda_device)
    for epoch in range(args.n_epoch):
        loss = train_epoch(args, epoch, model, criterion, optimizer, train_loader, train_dataset, Files, BVHWriters, characters, save_name) # target_characters
        wandb.log({"loss": loss})
        save(model, path + save_name, epoch)

else:
    epoch = 49
    load(model, path + save_name, epoch)
    # 프레임별로 입력으로 주기. 
    scores = []

    # test score
    # score = eval_epoch(args, epoch, model, test_loader, test_dataset, vocab)
    # scores.append(score)

