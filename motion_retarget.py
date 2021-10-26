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
def motion_collate_fn(inputs):

    # 인풋: (4,96,1,69,32) (캐릭터수, , 1, 조인트, 윈도우)

    input_motions, gt_motions = list(zip(*inputs)) # 모션안에 있는 window 갯수에 따라 다름.  # input_motions, output_motions
    
    # input = input_motions[0].unsqueeze(0)
    # for i in range(1, len(input_motions)):
    #     input = torch.cat((input, input_motions[i].unsqueeze(0)), dim=0)
        
    # gt = gt_motions[0].unsqueeze(0)
    # for i in range(1, len(gt_motions)):
    #     gt = torch.cat((gt, gt_motions[i].unsqueeze(0)), dim=0)


    input = torch.nn.utils.rnn.pad_sequence(input_motions, batch_first=True, padding_value=0)
    gt = torch.nn.utils.rnn.pad_sequence(gt_motions, batch_first=True, padding_value=0)

    batch = [
        input, 
        gt
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
print("cuda device: ", args.cuda_device)
log_path = os.path.join(args.save_dir, 'logs/')
wandb.init(project='transformer-retargeting', entity='loveyourdaddy')

print("cuda availiable: {}".format(torch.cuda.is_available()))
print("device: ", args.cuda_device)

""" Changable Parameters """
path = "./parameters/"
save_name = "211026_intra_retargeting_quaternion"# "211022_0_intra_retargeting_test_xyz/"

""" 1. load Motion Dataset """
characters = get_character_names(args)
dataset = create_dataset(args, characters)
# if args.is_train == 1:
#     batch_size = args.batch_size 
# else:
#     batch_size = 1 
loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=motion_collate_fn)
offsets = dataset.get_offsets()
print("characters:{}".format(characters))

""" 2.Set Learning Parameters  """
# args.num_joints = int(DoF/4) # 91 = 4 * 22 (+ position 3)
if args.add_offset:
    args.input_size = args.window_size + 1
else:
    args.input_size = args.window_size

""" 3. Train and Test  """
model = MotionGenerator(args, offsets) 
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
    
    # for every epoch 
    for epoch in range(args.n_epoch):
        fk_loss, loss = train_epoch(
            args, epoch, model, criterion, optimizer, 
            loader, dataset, 
            characters, save_name, Files)
        # wandb.log({"ele_loss": ele_loss})
        wandb.log({"fk_loss": fk_loss})
        wandb.log({"loss": loss})
        save(model, path + save_name, epoch)

else:
    epoch = 90
    load(model, path + save_name, epoch)
    eval_epoch(
        args, model, criterion, 
        dataset, loader, 
        characters, save_name, Files)