import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    # env
    parser.add_argument('--save_dir', type=str, default='./output/', help='directory for all savings')
    parser.add_argument('--cuda_device', type=str, default='cuda:0', help='cuda device number, eg:[cuda:0]')
    parser.add_argument('--is_train', type=int, default=1)
    parser.add_argument('--is_valid', type=int, default=0)
    parser.add_argument('--render', type=int, default=0)

    # learning parameter
    parser.add_argument('--learning_rate', type=float, default=1e-4 , help='lr')  # 2e-4 # 5e-5
    parser.add_argument('--weight_decay', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=0, help='penalty of sparsity')
    parser.add_argument('--activation', type=str, default='LeakyReLU', help='activation: ReLU, LeakyReLU, tanh')
    parser.add_argument('--n_epoch', type=int, default=10001)
    parser.add_argument('--epoch_begin', type=int, default=0)

    # Dataset representation
    parser.add_argument('--batch_size', type=int,default=32, help='batch_size')
    parser.add_argument('--rotation', type=str, default='quaternion', help='representatio0 of rotation:xyz, quaternion')
    parser.add_argument('--window_size', type=int, default=128, help='length of time axis per window')
    parser.add_argument('--num_motions', type=int, default=1)    
    # parser.add_argument('--input_size', type=int, default=0, help='')
    # parser.add_argument('--output_size', type=int, default=0, help='')
    # parser.add_argument('--n_enc_seq', type=int, default=0, help='')

    # Dataset representation (flag)
    parser.add_argument('--normalization', type=int, default=1)
    parser.add_argument('--root_pos_disp', type=int, default=0, help='represent root pos as displacement')
    parser.add_argument('--swap_dim', type=int, default=1,help='data_augment: 1 or 0')
    parser.add_argument('--add_offset', type=int, default=0, help='concat offset in dataset')
    parser.add_argument('--data_encoding', type=int, default=1, help='positional encoding')
    parser.add_argument('--data_augment', type=int, default=0, help='data_augment: 1 or 0')

    # Network
    parser.add_argument('--layer_norm_epsilon', type=float, default=1e-12)
    parser.add_argument('--i_pad', type=int, default=0)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--d_head', type=int, default=64)
    # joint 을 얼마나 줄일지에 대한 hidden dimension
    parser.add_argument('--d_hidn', type=int, default=91)
    # xyz embedding dimenstion: 69 -> (64) -> 32
    # quaternion embedding dimenstion: 91 -> 91 -> 111
    parser.add_argument('--embedding_dim', type=int, default=256,help='embedding dimension')  # window을 얼마나 줄일지에 대한 embedding
    parser.add_argument('--gan_mode', type=str, default='vanilla') # lsgan

    # loss 
    parser.add_argument('--rec_loss',     type=int, default=1, help='1. rec loss')
    parser.add_argument('--fk_loss',      type=int, default=0, help='1-2. fk loss')    
    parser.add_argument('--consist_loss', type=int, default=1, help='2. consistency loss')
    # parser.add_argument('--gan_loss',     type=int, default=0, help='3. gan loss')    
    # parser.add_argument('--reg_loss',     type=int, default=0, help='5. regularization loss on score(prob) matrix')
    # 3. latency consistenecy
    # 4. ee loss 

    return parser


def get_args():
    parser = get_parser()
    return parser.parse_args()


def get_std_bvh(args=None, dataset=None):
    if args is None and dataset is None:
        raise Exception('Unexpected parameter')
    if dataset is None:
        dataset = args.dataset
    std_bvh = './datasets/Mixamo/std_bvhs/{}.bvh'.format(dataset)
    return std_bvh


def get_test_std_bvh(args=None, dataset=None):
    if args is None and dataset is None:
        raise Exception('Unexpected parameter')
    if dataset is None:
        dataset = args.dataset
    std_bvh = './datasets/Mixamo/test_std_bvhs/{}.bvh'.format(dataset)
    return std_bvh


def try_mkdir(path):
    import os
    if not os.path.exists(path):
        os.system('mkdir -p {}'.format(path))
