import torch
import os
import option_parser
from datasets import get_character_names, create_dataset
from train import GeneralModel
from train import *
from test import *
from torch.utils.tensorboard import SummaryWriter


""" Set Env Parameters """
args = option_parser.get_args()
args.cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.n_topology = 2
para_path = "./parameters/"
print("cuda availiable: {}".format(torch.cuda.is_available()))
save_name = "220326_classify/"
# args.epoch_begin = 100
# args.is_train = False
log_dir = './run/' + save_name
writer = SummaryWriter(log_dir, flush_secs=1)

""" load Motion Dataset """
characters = get_character_names(args)
dataset = create_dataset(args, characters)
loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False) 
print("characters:{}".format(characters))

""" load model  """
general_model = GeneralModel(args, characters, dataset) 

""" Load model if load mode """
if args.epoch_begin:
    general_model.load(para_path+save_name, args.epoch_begin)
if args.is_train == True:
    # for every epoch
    for epoch in range(args.epoch_begin, args.n_epoch):
        general_model.train_epoch(epoch, loader, save_name)
        # rec_losses, rec_loss1, rec_loss2, rec_loss3 = general_model.train_epoch(epoch, loader, save_name)
        
        writer.add_scalar("Loss/rec_loss",  np.mean(general_model.rec_losses),  epoch)
        writer.add_scalar("Loss/rec_loss1", np.mean(general_model.rec_losses1), epoch)
        writer.add_scalar("Loss/rec_loss2", np.mean(general_model.rec_losses2), epoch)
        writer.add_scalar("Loss/rec_loss3", np.mean(general_model.rec_losses3), epoch)

        if epoch % 100 == 0:
            general_model.save(para_path + save_name, epoch)
else:
    # only test losses 
    general_model.eval_epoch(args.epoch_begin, loader, dataset, characters, save_name)
