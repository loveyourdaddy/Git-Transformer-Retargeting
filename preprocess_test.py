import os
import numpy as np
import copy
from datasets.bvh_parser import BVH_file
from datasets.motion_dataset import MotionData
from option_parser import get_args, try_mkdir

def collect_bvh(data_path, character, files):
    print('begin {}'.format(character))
    motions = []

    for i, motion in enumerate(files):
        if not os.path.exists(data_path + character + '/test/' + motion):
            continue
        file = BVH_file(data_path + character + '/test/' + motion)
        new_motion = file.to_tensor().permute((1, 0)).numpy()
        motions.append(new_motion)
    
    # (112, frames (differnet for motions), 84)
    save_file = data_path + character + '_test' + '.npy'

    np.save(save_file, motions)
    print('Npy file saved at {}'.format(save_file))


if __name__ == '__main__':
    prefix = './datasets/Mixamo/'

    characters = [f for f in os.listdir(prefix) if os.path.isdir(os.path.join(prefix, f))]
    if 'std_bvhs' in characters: characters.remove('std_bvhs')
    if 'mean_var' in characters: characters.remove('mean_var')
    if 'std_bvhs_test' in characters: characters.remove('std_bvhs_test')
    if 'mean_var_test' in characters: characters.remove('mean_var_test')

    try_mkdir(os.path.join(prefix, 'std_bvhs_test'))
    try_mkdir(os.path.join(prefix, 'mean_var_test'))

    for character in characters:
        data_path = os.path.join(prefix, character) + '/test'
        files = sorted([f for f in os.listdir(data_path) if f.endswith(".bvh")])

        collect_bvh(prefix, character, files) # motion 파일들 npy 파일에 저장
        # copy_std_bvh(prefix, character, files) #  0번째 파일을 /std_bvh 폴더에 저장함
        # write_statistics(character, './datasets/Mixamo/mean_var_test/') # std 파일의 normalization data을 저장함  
