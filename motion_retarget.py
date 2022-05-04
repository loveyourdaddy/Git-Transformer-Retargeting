import torch
import numpy as np
import os
import option_parser
from datasets import get_character_names, create_dataset
from general_model import *
from test import *
from torch.utils.tensorboard import SummaryWriter

""" Set Parameters """
args = option_parser.get_args()
save_name = "220501_total_loss0/"  # 220501_smaller_fk
args.epoch_begin = 800
args.is_train = False

""" Set Env """
args.cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.n_topology = 2
print("cuda availiable: {}".format(torch.cuda.is_available()))
para_path = "./parameters/" + save_name  # "/media/Personal/InseoJang/"
log_dir = "./run/" + save_name
try_mkdir(log_dir)
writer = SummaryWriter(log_dir, flush_secs=1)

""" load Motion Dataset """
characters = get_character_names(args)
dataset = create_dataset(args, characters)
loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=False)
print("characters:{}".format(characters))

""" load model  """
general_model = GeneralModel(
    args, characters, dataset)

""" Load model if load mode """
if args.epoch_begin:
    general_model.load(para_path, args.epoch_begin)

if args.is_train == True:
    # for every epoch
    for epoch in range(args.epoch_begin, args.n_epoch):
        general_model.train_epoch(epoch, loader, save_name)

        writer.add_scalar("Loss/element_loss",
                          np.mean(general_model.element_losses), epoch)
        writer.add_scalar("Loss/cross_loss",
                          np.mean(general_model.cross_losses), epoch)

        writer.add_scalar("Loss/smooth_loss",
                          np.mean(general_model.smooth_losses), epoch)
        writer.add_scalar("Loss/fk_loss",
                          np.mean(general_model.fk_losses), epoch)

        writer.add_scalar("Loss/latent_loss",
                          np.mean(general_model.latent_losses), epoch)
        writer.add_scalar("Loss/cycle_loss",
                          np.mean(general_model.cycle_losses), epoch)

        writer.add_scalar("Loss/ee_loss",
                          np.mean(general_model.ee_losses), epoch)

        writer.add_scalar("Loss/root_loss",
                          np.mean(general_model.root_losses), epoch)
        writer.add_scalar("Loss/root_rotation_loss",
                          np.mean(general_model.root_rotation_losses), epoch)

        writer.add_scalar("Loss/GAN_G_fake_loss",
                          np.mean(general_model.G_fake_losses), epoch)
        writer.add_scalar("Loss/GAN_D_real_loss",
                          np.mean(general_model.D_real_losses), epoch)
        writer.add_scalar("Loss/GAN_D_fake_loss",
                          np.mean(general_model.D_fake_losses), epoch)

        if epoch % args.save_epoch == 0 and epoch != 0:
            general_model.save(para_path, epoch)
else:
    # only test losses
    general_model.eval_epoch(args.epoch_begin, dataset, save_name)
