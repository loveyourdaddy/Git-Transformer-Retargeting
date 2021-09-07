import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    
    # env 
    parser.add_argument('--save_dir', type=str, default='./output/', help='directory for all savings')
    parser.add_argument('--cuda_device', type=str, default='cuda:0', help='cuda device number, eg:[cuda:0]')

    # learning parameter 
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate') # 2e-4 # 5e-5
    parser.add_argument('--alpha', type=float, default=0, help='penalty of sparsity')
    parser.add_argument('--upsampling', type=str, default='linear', help="'stride2' or 'nearest', 'linear'")
    parser.add_argument('--downsampling', type=str, default='stride2', help='stride2 or max_pooling')
    # parser.add_argument('--batch_normalization', type=int, default=0, help='batch_norm: 1 or 0')
    parser.add_argument('--activation', type=str, default='LeakyReLU', help='activation: ReLU, LeakyReLU, tanh')
    parser.add_argument('--rotation', type=str, default='quaternion', help='representation of rotation:euler_angle, quaternion')
    parser.add_argument('--data_augment', type=int, default=0, help='data_augment: 1 or 0') # 1
    parser.add_argument('--n_epoch', type=int, default=1000) # duplicated 

    # Dataset
    parser.add_argument('--normalization', type=int, default = 1)
    parser.add_argument('--position_encoding', type=int, default = 0, help='positional encoding')
    parser.add_argument('--root_pos_disp', type=int, default = 1, help='represent root pos as displacement')
    parser.add_argument('--num_motions', type=int, default=1) # _for_character
    parser.add_argument('--window_size', type=int, default=128, help='length of time axis per window')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
   
    # Network
    parser.add_argument('--d_hidn', type=int, default=64) # embedding dimenstion: 256
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers')

    # Training
    parser.add_argument('--root_weight', type=int, default=1, help='weighted loss for root displacement')

    # 필요없는건 지웁시다 from skeleton aware  
    parser.add_argument('--kernel_size', type=int, default=15, help='must be odd')
    parser.add_argument('--base_channel_num', type=int, default=-1)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--skeleton_dist', type=int, default=2)
    parser.add_argument('--skeleton_pool', type=str, default='mean')
    parser.add_argument('--extra_conv', type=int, default=0)
    parser.add_argument('--padding_mode', type=str, default='reflection')
    parser.add_argument('--dataset', type=str, default='Mixamo')
    parser.add_argument('--fk_world', type=int, default=0)
    parser.add_argument('--patch_gan', type=int, default=1)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--skeleton_info', type=str, default='concat')
    parser.add_argument('--ee_loss_fact', type=str, default='height')
    parser.add_argument('--pos_repr', type=str, default='3d')
    parser.add_argument('--D_global_velo', type=int, default=0)
    parser.add_argument('--gan_mode', type=str, default='lsgan')
    parser.add_argument('--pool_size', type=int, default=50)
    parser.add_argument('--is_train', type=int, default=1)
    parser.add_argument('--is_valid', type=int, default=0)    
    parser.add_argument('--model', type=str, default='mul_top_mul_ske')
    parser.add_argument('--epoch_begin', type=int, default=0)
    parser.add_argument('--lambda_rec', type=float, default=5)
    parser.add_argument('--lambda_cycle', type=float, default=5)
    parser.add_argument('--lambda_ee', type=float, default=100)
    parser.add_argument('--lambda_global_pose', type=float, default=2.5)
    parser.add_argument('--lambda_position', type=float, default=1)
    parser.add_argument('--ee_velo', type=int, default=1)
    parser.add_argument('--ee_from_root', type=int, default=1)
    parser.add_argument('--scheduler', type=str, default='none')
    parser.add_argument('--rec_loss_mode', type=str, default='extra_global_pos')
    parser.add_argument('--adaptive_ee', type=int, default=0)
    parser.add_argument('--simple_operator', type=int, default=0)
    parser.add_argument('--use_sep_ee', type=int, default=0)
    parser.add_argument('--eval_seq', type=int, default=0)

    # check it needed - from 언어모델
    parser.add_argument('--n_enc_seq', type=int, default=256)
    parser.add_argument('--n_dec_seq', type=int, default=256)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--d_head', type=int, default=64)
    parser.add_argument('--layer_norm_epsilon', type=float, default=1e-12)
    parser.add_argument('--n_output', type=int, default=0)
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


def try_mkdir(path):
    import os
    if not os.path.exists(path):
        os.system('mkdir -p {}'.format(path))
