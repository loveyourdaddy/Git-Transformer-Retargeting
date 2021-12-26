import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    
    # env 
    parser.add_argument('--save_dir', type=str, default='./output/', help='directory for all savings')
    parser.add_argument('--cuda_device', type=str, default='cuda:0', help='cuda device number, eg:[cuda:0]')
    parser.add_argument('--is_train', type=int, default=1)
    parser.add_argument('--is_valid', type=int, default=0)
    parser.add_argument('--render', type=int, default=0)

    # options
    parser.add_argument('--normalization', type=int, default = 1)
    parser.add_argument('--add_offset', type=int, default=0, help='concat offset in dataset')
    parser.add_argument('--position_encoding', type=int, default = 0, help='positional encoding')
    parser.add_argument('--root_pos_disp', type=int, default = 0, help='represent root pos as displacement')
    parser.add_argument('--data_augment', type=int, default=0, help='data_augment: 1 or 0') 
    parser.add_argument('--input_size', type=int, default=0, help='') 
    parser.add_argument('--output_size', type=int, default=0, help='')
    parser.add_argument('--weight_root_loss', type=int, default=0, help='flag for weight_root_loss')
    parser.add_argument('--fk_loss', type=int, default=0, help='fk los')

    
    # learning parameter 
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate') # 2e-4 # 5e-5
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size') # 64
    parser.add_argument('--weight_decay', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=0, help='penalty of sparsity')
    parser.add_argument('--activation', type=str, default='LeakyReLU', help='activation: ReLU, LeakyReLU, tanh')
    parser.add_argument('--n_epoch', type=int, default=1001) # duplicated 
    parser.add_argument('--epoch_begin', type=int, default=0)
    # parser.add_argument('--upsampling', type=str, default='linear', help="'stride2' or 'nearest', 'linear'")
    # parser.add_argument('--downsampling', type=str, default='stride2', help='stride2 or max_pooling')
    # parser.add_argument('--batch_normalization', type=int, default=0, help='batch_norm: 1 or 0')

    # Dataset representation & option
    parser.add_argument('--rotation', type=str, default='xyz', help='representatio0 of rotation:xyz, quaternion')
    parser.add_argument('--root_weight', type=int, default=10, help='weighted loss for root displacement')
    parser.add_argument('--window_size', type=int, default=128, help='length of time axis per window')
    parser.add_argument('--num_motions', type=int, default=1) # num motions for_character. dummy value 1 

        
    # Network
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--d_hidn', type=int, default=64) # embedding dimenstion: 256
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--d_head', type=int, default=64)
    parser.add_argument('--layer_norm_epsilon', type=float, default=1e-12)
    parser.add_argument('--i_pad', type=int, default=0)

    return parser


def get_args():
    parser = get_parser()
    return parser.parse_args()


def get_std_bvh(args=None, dataset=None):
    if args is None and dataset is None: raise Exception('Unexpected parameter')
    if dataset is None: dataset = args.dataset
    std_bvh = './datasets/Mixamo/std_bvhs/{}.bvh'.format(dataset)
    return std_bvh

def get_test_std_bvh(args=None, dataset=None):
    if args is None and dataset is None: raise Exception('Unexpected parameter')
    if dataset is None: dataset = args.dataset
    std_bvh = './datasets/Mixamo/test_std_bvhs/{}.bvh'.format(dataset)
    return std_bvh

def try_mkdir(path):
    import os
    if not os.path.exists(path):
        os.system('mkdir -p {}'.format(path))
