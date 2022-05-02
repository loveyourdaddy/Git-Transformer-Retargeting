import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    # env
    parser.add_argument('--save_epoch', type=int, default=100)
    parser.add_argument('--writing_epoch', type=int, default=200)
    parser.add_argument('--save_dir', type=str,
                        default='./output/', help='directory for all savings')
    parser.add_argument('--cuda_device', type=str,
                        default='cuda:0', help='cuda device number, eg:[cuda:0]')
    parser.add_argument('--is_train', type=bool, default=True)
    parser.add_argument('--is_valid', type=int, default=0)
    parser.add_argument('--render', type=int, default=0)

    # learning parameter
    parser.add_argument('--learning_rate', type=float,
                        default=1e-4, help='lr')
    parser.add_argument('--n_epoch', type=int, default=10001)
    parser.add_argument('--weight_decay', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=0,
                        help='penalty of sparsity')
    parser.add_argument('--activation', type=str, default='LeakyReLU',
                        help='activation: ReLU, LeakyReLU, tanh')
    parser.add_argument('--epoch_begin', type=int, default=0)

    # Dataset representation
    parser.add_argument('--batch_size', type=int,
                        default=4, help='batch_size')  # 32
    parser.add_argument('--rotation', type=str, default='quaternion',
                        help='representatio0 of rotation:xyz, quaternion')
    parser.add_argument('--window_size', type=int, default=128,
                        help='length of time axis per window')
    parser.add_argument('--num_motions', type=int, default=1)
    parser.add_argument('--ee_velo', type=int, default=1)
    parser.add_argument('--ee_from_root', type=int, default=1)

    # Dataset representation (flag)
    parser.add_argument('--normalization', type=int, default=1)
    parser.add_argument('--root_pos_as_disp', type=int,
                        default=1, help='represent root pos as displacement')
    parser.add_argument('--swap_dim', type=int, default=1,
                        help='Data swap: 1 or 0')
    parser.add_argument('--add_offset', type=int, default=0,
                        help='concat offset in dataset')
    parser.add_argument('--data_encoding', type=int,
                        default=1, help='positional encoding')
    parser.add_argument('--data_augment', type=int,
                        default=0, help='data_augment: 1 or 0')

    # Network
    parser.add_argument('--layer_norm_epsilon', type=float, default=1e-12)
    parser.add_argument('--i_pad', type=int, default=0)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--d_head', type=int, default=64)
    parser.add_argument('--d_hidn', type=int, default=512)
    parser.add_argument('--gan_mode', type=str, default='lsgan')  # vanilla

    # SAN Structure weight
    # parser.add_argument('--kernel_size', type=int, default='15')
    # parser.add_argument('--num_layers', type=int, default='2')
    # parser.add_argument('--skeleton_dist', type=int, default='2')
    # parser.add_argument('--padding_mode', type=str, default='reflect')
    # parser.add_argument('--skeleton_pool', type=str, default='mean')
    # parser.add_argument('--skeleton_info', type=str, default='concat') # ?
    # parser.add_argument('--upsampling', type=str, default='linear', help="'stride2' or 'nearest', 'linear'")
    # parser.add_argument('--pos_repr', type=str, default='3d')

    # loss
    parser.add_argument('--rec_loss', type=int, default=1, help='1. rec loss')
    parser.add_argument('--fk_loss',  type=int, default=1, help='1-2. fk loss')
    parser.add_argument('--ltc_loss', type=int, default=1,
                        help='2. consistency loss')
    parser.add_argument('--cyc_loss', type=int,
                        default=1, help='3. cycle loss')
    # parser.add_argument('--gan_loss', type=int, default=1, help='3. gan loss')
    # parser.add_argument('--reg_loss', type=int, default=0, help='5. regularization loss on score(prob) matrix')
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
