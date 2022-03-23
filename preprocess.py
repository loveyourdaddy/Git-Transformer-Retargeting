import os
import numpy as np
import copy
from datasets.bvh_parser import BVH_file
from datasets.motion_dataset import MotionData
# from option_parser import get_args, try_mkdir
from option_parser import *
import option_parser

def collect_bvh(args, data_path, character, files):
    print('begin {}'.format(character))

    # save motion as .npy file 
    motions = []
    
    for i, motion in enumerate(files):
        if args.is_train == 1:
            path = data_path + character + '/' + motion
        elif args.is_train == 0: 
            path = data_path + character + '/test/' + motion
        else:
            print("error")

        if not os.path.exists(path):
            print("no data")
            continue
        file = BVH_file(path)
        new_motion = file.to_tensor().permute((1, 0)).numpy()
        motions.append(new_motion)

    if args.is_train == 1:
        save_name = data_path + character + '.npy'
    elif args.is_train == 0:
        save_name = data_path + character + '_test.npy'
    else:
        print("error")
    np.save(save_name, motions)

    """ Get body part index and save it"""
    # Get body part index
    path = data_path + character + '/' + motion
    file = BVH_file(path)
    body_part_index = file.get_body_part_index()
    
    # Save it 
    save_name = data_path + '/body_part_index/' + character + '.npy'
    np.save(save_name, body_part_index)

    print('Npy file saved at {}'.format(save_name))

def copy_std_bvh(args, data_path, character, files):
    """
    copy an arbitrary bvh file as a static information (skeleton's offset) reference
    """
    if args.is_train == 1:
        cmd = 'cp \"{}\" ./datasets/Mixamo/std_bvhs/{}.bvh'.format(data_path + character + '/' + files[0], character)
    elif args.is_train == 0:
        cmd = 'cp \"{}\" ./datasets/Mixamo/std_bvhs/{}.bvh'.format(data_path + character + '/test/' + files[0], character)
    else:
        print("error")
    os.system(cmd)

# mean / var 저장
def write_statistics(args, character, path):
    # args = option_parser.get_args()
    # new_args = copy.copy(args)
    args.data_augment = 0
    args.dataset = character

    dataset = MotionData(args, 1)

    mean = dataset.mean
    var = dataset.var
    mean = mean.cpu().numpy()[0, ...]
    var = var.cpu().numpy()[0, ...]

    if args.is_train == 1:
        np.save(path + '{}_mean.npy'.format(character), mean)
        np.save(path + '{}_var.npy'.format(character), var)
    elif args.is_train == 0:
        np.save(path + '{}_mean_test.npy'.format(character), mean)
        np.save(path + '{}_var_test.npy'.format(character), var)
    else: 
        print("error")


if __name__ == '__main__':
    args = option_parser.get_args()

    prefix = './datasets/Mixamo/'
    characters = [f for f in os.listdir(prefix) if os.path.isdir(os.path.join(prefix, f))]
    if 'std_bvhs' in characters: characters.remove('std_bvhs')
    if 'mean_var' in characters: characters.remove('mean_var')
    if 'body_part_index' in characters: characters.remove('body_part_index')

    try_mkdir(os.path.join(prefix, 'std_bvhs'))
    try_mkdir(os.path.join(prefix, 'mean_var'))

    for character in characters:
        if args.is_train == 1:
            data_path = os.path.join(prefix, character)
        elif args.is_train == 0:
            data_path = os.path.join(prefix, character) + '/test' # /validation
        else:
            print("Error")

        files = sorted([f for f in os.listdir(data_path) if f.endswith(".bvh")])
        collect_bvh(args, prefix, character, files)
        copy_std_bvh(args, prefix, character, files)
        write_statistics(args, character, './datasets/Mixamo/mean_var/')
   