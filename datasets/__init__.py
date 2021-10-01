def get_character_names(args):
    if args.is_train:
        """
        Train Case
        Put the name of subdirectory in retargeting/datasets/Mixamo as [[names of group A], [names of group B]]
        """
        characters = [['Aj', 'Aj', 'Aj', 'Aj'], ['Aj', 'Aj', 'Aj', 'Aj']]
        # characters = [['Aj', 'BigVegas', 'Kaya', 'SportyGranny'], ['Aj', 'BigVegas', 'Kaya', 'SportyGranny']]
        
        # characters = [['Aj', 'BigVegas', 'Kaya', 'SportyGranny'], ['Mousey_m', 'Goblin_m', 'Mremireh_m', 'Vampire_m']]                          
        # characters = [['Aj', 'BigVegas', 'Kaya', 'SportyGranny'], ['SMPL', 'SMPL', 'SMPL', 'SMPL']]

    else:
        """
        Test Case
        To run evaluation successfully, number of characters in both groups must be the same. Repeat is okay.
        """ 
        characters = [['Aj', 'BigVegas', 'Kaya', 'SportyGranny'], ['Aj', 'BigVegas', 'Kaya', 'SportyGranny']]                            
        # characters = [['Aj', 'BigVegas', 'Kaya', 'SportyGranny'], ['Mousey_m', 'Goblin_m', 'Mremireh_m', 'Vampire_m']]        
        # characters = [['Aj', 'BigVegas', 'Kaya', 'SportyGranny'], ['SMPL', 'SMPL', 'SMPL', 'SMPL']]
        # characters = [['Mousey_m', 'Goblin_m', 'Mremireh_m', 'Vampire_m'], ['SMPL', 'SMPL', 'SMPL', 'SMPL']]
        
        # check eval_seq
        tmp = characters[1][args.eval_seq]
        characters[1][args.eval_seq] = characters[1][0]
        characters[1][0] = tmp

    return characters


def create_dataset(args, character_names=None):
    from datasets.combined_motion import TestData, MixedData, ValidationData
    return MixedData(args, character_names)
    # if args.is_train:
    #     return MixedData(args, character_names)
    # elif args.is_valid:
    #     return ValidationData(args, character_names)
    # else:
    #     return TestData(args, character_names)


def get_test_set():
    with open('./datasets/test_list.txt', 'r') as file:
        list = file.readlines()
        list = [f[:-1] for f in list]
        # print("list:{}".format(list))

        return list


def get_validation_set():
    with open('./datasets/validation_list.txt', 'r') as file:
        list = file.readlines()
        list = [f[:-1] for f in list]
        # print("list:{}".format(list))

        return list


def get_train_list():
    with open('./datasets/train_list.txt', 'r') as file:
        list = file.readlines()
        list = [f[:-1] for f in list]
        return list
