# import json
import torch
import os
from datasets import bvh_writer
import option_parser
from datasets import get_character_names, create_dataset
from model import MotionGenerator
from datasets.bvh_parser import BVH_file
from datasets.bvh_writer import BVH_writer
import wandb
from train import *

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
args_ = option_parser.get_args()
args = args_
args.cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_path = os.path.join(args.save_dir, 'logs/')
wandb.init(project='transformer-retargeting', entity='loveyourdaddy')

print("cuda availiable: {}".format(torch.cuda.is_available()))
print("device: ", args.cuda_device)

""" Changable Parameters """
# args.is_train = False 
path = "./parameters/"
save_name = "211002_transformer1_with_fk_3e-3/"

""" 1. load Motion Dataset """
characters = get_character_names(args)
train_dataset = create_dataset(args, characters)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=motion_collate_fn)
print("characters:{}".format(characters))

""" 2.Set Learning Parameters  """
DoF = train_dataset.GetDoF()
args.num_joints = int(DoF/4) # 91 = 4 * 22 (+ position 3)
# args.num_motions = len(train_dataset) / 4

""" 3. Train and Test  """
model = MotionGenerator(args, characters, train_dataset)
model.to(args.cuda_device)
wandb.watch(model)

criterion = torch.nn.MSELoss() # torch.nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay = args.weight_decay) 

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

if args.is_train == 1:
    print("cuda device: ", args.cuda_device)
    for epoch in range(args.n_epoch):
        ele_loss, fk_loss, loss = train_epoch(args, epoch, model, criterion, optimizer, train_loader, train_dataset, Files, BVHWriters, characters, save_name) # target_characters
        wandb.log({"ele_loss": ele_loss})
        wandb.log({"fk_loss": fk_loss})
        wandb.log({"loss": loss})
        save(model, path + save_name, epoch)

else:
    epoch = 10
    load(model, path + save_name, epoch)
    eval_epoch(args, model, train_loader)
    # 프레임별로 입력으로 주기. 
    scores = []

    # test score
    # score = eval_epoch(args, epoch, model, test_loader, test_dataset, vocab)
    # scores.append(score)

