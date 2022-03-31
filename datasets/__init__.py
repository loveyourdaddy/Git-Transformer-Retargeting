def get_character_names(args):

    # intra strucutre retargeting    
    # characters = [['Aj', 'Aj', 'Aj', 'Aj'], ['Aj', 'Aj', 'Aj', 'Aj']]
    # characters = [['Aj', 'Aj', 'Aj', 'Aj'], ['Aj', 'BigVegas', 'Kaya', 'SportyGranny']]    

    # cross strucutre retargeting
    characters = [['BigVegas', 'BigVegas', 'BigVegas', 'BigVegas'],  ['Mousey_m', 'Goblin_m', 'Mremireh_m', 'Vampire_m']]
    # characters = [['Aj', 'BigVegas', 'Kaya', 'SportyGranny'], ['Mousey_m', 'Goblin_m', 'Mremireh_m', 'Vampire_m']]

    return characters


def create_dataset(args, character_names=None):
    from datasets.combined_motion import TestData, MixedData

    if args.is_train:
        return MixedData(args, character_names)
    else:
        return TestData(args, character_names)

def get_test_set():
    with open('./datasets/test_list.txt', 'r') as file:
        list = file.readlines()
        list = [f[:-1] for f in list]

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
