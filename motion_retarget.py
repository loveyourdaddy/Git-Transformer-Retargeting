import torch
import os
from datasets import bvh_writer
import option_parser
from datasets import get_character_names, create_dataset
from model import MotionGenerator
from model import Discriminator
from datasets.bvh_parser import BVH_file
from datasets.bvh_writer import BVH_writer
# import wandb
from train import *
from test import *
from torch.utils.tensorboard import SummaryWriter

""" motion data collate function """
def motion_collate_fn(inputs):
    # Data foramt: (4,96,1,69,32) (캐릭터수, , 1, 조인트, 윈도우)
    source_motions, target_motions = list(zip(*inputs)) #enc_input_motions, 

    source_motions = torch.nn.utils.rnn.pad_sequence(
        source_motions, batch_first=True, padding_value=0)
    target_motions = torch.nn.utils.rnn.pad_sequence(
        target_motions, batch_first=True, padding_value=0)
    # gt = torch.nn.utils.rnn.pad_sequence(
    #     gt_motions, batch_first=True, padding_value=0)

    batch = [
        source_motions,
        target_motions 
    ]

    return batch

# def save(model, path, epoch):
#     try_mkdir(path)
#     path = os.path.join(path, str(epoch))
#     torch.save(model.state_dict(), path)

def save(model, optimizer, path, name, epoch, i):
    try_mkdir(path)
    path_para = os.path.join(path, name + str(i) +'_' + str(epoch))
    torch.save(model.state_dict(), path_para)

    path_para = os.path.join(path, name + str(i)+ 'Opti_' + str(epoch))
    torch.save(optimizer.state_dict(), path_para)

# def load(model, path, epoch):
#     path = os.path.join(path, str(epoch))

#     if not os.path.exists(path):
#         raise Exception('Unknown loading path')
#     model.load_state_dict(torch.load(path))
#     print('load succeed')

def load(model, optimizer, path, name, epoch):
    path_para = os.path.join(path, name + str(epoch))
    
    if not os.path.exists(path_para):
        raise Exception('Unknown loading path')
    model.load_state_dict(torch.load(path_para))

    path_para = os.path.join(path, name + 'Opti' + str(epoch))
    if not os.path.exists(path_para):
        raise Exception('Unknown loading path')
    optimizer.load_state_dict(torch.load(path_para))

    print('load succeed')


""" Set Env Parameters """
args = option_parser.get_args()
args.cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.n_topology = 2
para_path = "./parameters/"
print("cuda availiable: {}".format(torch.cuda.is_available()))
save_name = "220204_0_rec_fk_consistency_loss/"
log_dir = './run/' + save_name
writer = SummaryWriter(log_dir, flush_secs=1)

""" load Motion Dataset """
characters = get_character_names(args)
dataset = create_dataset(args, characters)
loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=False, collate_fn=motion_collate_fn)
offsets = dataset.get_offsets()
print("characters:{}".format(characters))

""" load model  """
generator_models = []
discriminator_models = []
optimizerGs = []
optimizerDs = []
for i in range(args.n_topology):
    # model 
    generator_model     = MotionGenerator(args, offsets, i)
    discriminator_model = Discriminator(args, offsets, i)    
    generator_model.to(args.cuda_device)
    discriminator_model.to(args.cuda_device)

    # optimizer
    optimizerG = torch.optim.Adam(generator_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizerD = torch.optim.Adam(discriminator_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # add to list 
    generator_models.append(generator_model)
    discriminator_models.append(discriminator_model)

    optimizerGs.append(optimizerG)
    optimizerDs.append(optimizerD)

# wandb.watch(generator_models[0], log="all") # , log_graph=True
# wandb.watch(discriminator_models[0], log="all") # , log_graph=True

""" Set BVH writers """ 
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

""" Load model if load mode """
# args.epoch_begin = 500
if args.epoch_begin:
    for i in range(args.n_topology):
        load(generator_models[i],     optimizerGs[i], para_path+save_name, "Gen"+i+"_", args.epoch_begin)
        load(discriminator_models[i], optimizerDs[i], para_path+save_name, "Dis"+i+"_", args.epoch_begin)

if args.is_train == 1:
    # for every epoch
    for epoch in range(args.epoch_begin, args.n_epoch):
        rec_loss0, rec_loss1, fk_losses, G_loss, D_loss_real, D_loss_fake, consist_loss = train_epoch(
            args, epoch, generator_models, discriminator_models, optimizerGs, optimizerDs,
            loader, dataset, characters, save_name, Files)

        writer.add_scalar("Loss/rec_loss0", rec_loss0, epoch)
        writer.add_scalar("Loss/rec_loss1", rec_loss1, epoch)
        writer.add_scalar("Loss/fk_loss", fk_losses, epoch)
        writer.add_scalar("Loss/G_loss", G_loss, epoch)
        writer.add_scalar("Loss/D_loss_real", D_loss_real, epoch)
        writer.add_scalar("Loss/D_loss_fake", D_loss_fake, epoch)
        writer.add_scalar("Loss/consist_loss", consist_loss, epoch)
        
        # wandb.log({"loss": rec_loss},           step=epoch)
        # wandb.log({"fk_loss": fk_loss},         step=epoch)
        # wandb.log({"G_loss": G_loss},           step=epoch)
        # wandb.log({"D_loss_real": D_loss_real}, step=epoch)
        # wandb.log({"D_loss_fake": D_loss_fake}, step=epoch)

        if epoch % 100 == 0:
            for i in range(args.n_topology):
                save(generator_models[i], optimizerGs[i], para_path + save_name, "Gen", epoch, i)
                save(discriminator_models[i], optimizerDs[i], para_path + save_name, "Dis", epoch, i)

else:
    epoch = 30

    load(generator_model, para_path+save_name, "Gen", args.epoch_begin)
    load(discriminator_model, para_path+save_name, "Dis", args.epoch_begin)

    # only test losses 
    eval_epoch(args, generator_model, discriminator_model, dataset, loader,
        characters, save_name, Files)
