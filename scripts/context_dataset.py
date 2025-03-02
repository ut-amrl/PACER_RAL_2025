import torch
from torch.utils.data import Dataset, DataLoader
from utils import img_to_tensor_format, viz_contexts
import pytorch_lightning as pl
from termcolor import cprint
from copy import deepcopy
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import random
import pickle
import math
import yaml
import glob
import os

# Add the top-level project directory to the Python path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


random.seed(42)
IMG_WIDTH = 32

# example preference ordering:
# red_brick: 0
# yellow_brick: 0
# concrete: 0
# pebble_pavement: 1
# mulch: 2
# marble_rock: 3
# grass: 4
# bush: 5


class ContextDataset(Dataset):
    def __init__(self, train, pref_config, context_data_config_path, length = 1000):
        pref_config = pref_config
        self.data_map = [] # preferences, 3-tuple
        self.patch_bank = {}
        self.terrains = set()
        self.len = 0
        self.total = 0
        self.train = train

        cprint('Reading the yaml file at : {}'.format(context_data_config_path), 'green')
        self.context_data_config = yaml.load(open(context_data_config_path, 'r'), Loader=yaml.FullLoader)

        pref_config = yaml.load(open(pref_config, 'r'), Loader=yaml.FullLoader)
        self.config_to_pref_list(pref_config)
        self.load_data()

        self.num_preferences = len(self.data_map)
        cprint('Context dataset length: {}'.format(self.len), 'green')
        cprint('Context dataset number of preferences: {}'.format(self.num_preferences), 'green')


    def config_to_pref_list(self, pref_config):
        '''
            populate self.data_map
            self.data_map[i] is a list of context combinations for ordering i
        '''
        pref_set = pref_config['preferences']
        for ordering in pref_set:
            self.terrains.update(ordering.keys())
            preference = {}
            for key in ordering:
                preference[ordering[key]] = preference.get(ordering[key], []) + [key]
            preference = dict(sorted(preference.items()))
            preference_combos = self.create_preference_combos(preference)
            random.shuffle(preference_combos)

            # if train, take the top 80% of the ans
            l = len(preference_combos)
            if self.train:
                preference_combos = preference_combos[:math.ceil(0.8 * l)]
            else:
                preference_combos = preference_combos[math.ceil(0.8 * l):]
            self.data_map.append(preference_combos)
            self.len += len(preference_combos)

    def create_preference_combos(self, preference):
        '''
            create a list of all possible 3 preference orderings given a preference list
            returns a list of 3 preference lists
        '''
        def backtrack(cur_matrix, pairs, index, ans):
            if len(cur_matrix) == 3:
                ans.append(deepcopy(cur_matrix))
            else:
                for i in range(index, len(pairs)):
                    cur_matrix.append(pairs[i])
                    backtrack(cur_matrix, pairs, i + 1, ans)
                    cur_matrix.pop()

        # 1 == 2 == 3 < 4 < 5 < 6 < 7
        # preference = [[1,2,3],[4],[5],[6],[7]]
        pairs = []
        for i in range(len(preference)):
            for j in range(i + 1, len(preference)):
                for el1 in preference[i]:
                    for el2 in preference[j]:
                        pairs.append([el1, el2])

        cur_matrix = []
        ans = []

        backtrack(cur_matrix, pairs, 0, ans)
        return ans
    
    def load_data(self):
        if self.train:
            file_roots = [pickle_files_root for pickle_files_root in self.context_data_config['train']]
        else:
            file_roots = [pickle_files_root for pickle_files_root in self.context_data_config['val']]

        for root in tqdm(file_roots, desc="Loading Context Data"):
            if 'dataset_root_dir' in self.context_data_config:
                paths = glob.glob(os.path.join(self.context_data_config['dataset_root_dir'], os.path.join(root, '*.pkl')))
            
            label = root.split('/')[-2]
            if label not in self.terrains:
                continue
            l = self.patch_bank.get(label, [])
            for path in paths:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                l.append(data['patches'])
            self.patch_bank[label] = l
        cprint('Patch Bank labels', 'green')
        for key in self.patch_bank.keys():
            cprint('{}: {}'.format(key, len(self.patch_bank[key])), 'green')

    # ex of example: [['red_brick', 'yellow_brick'], ['concrete', 'pebble_pavement'], ['mulch', 'marble_rock']]
    def get_sample(self, example):
        arr = np.empty((9, IMG_WIDTH * 2, IMG_WIDTH))

        for i in range(len(example)):
            label_pair = example[i]

            rand1 = random.randint(0, len(self.patch_bank[label_pair[0]])-1)
            rand2 = random.randint(0, len(self.patch_bank[label_pair[1]])-1)
            img1 = self.patch_bank[label_pair[0]][rand1][0]
            img2 = self.patch_bank[label_pair[1]][rand2][0]

            # iterate until the images has less than 5% black pixels
            j = 0
            while (j < len(self.patch_bank[label_pair[0]][rand1]) and np.sum(img1 == 0) > 0.05 * IMG_WIDTH * 2 * IMG_WIDTH * 3):
                img1 = self.patch_bank[label_pair[0]][rand1][j]
                j +=1

            j = 0
            while (j < len(self.patch_bank[label_pair[1]][rand2]) and np.sum(img2 == 0) > 0.05 * IMG_WIDTH * 2 * IMG_WIDTH * 3):
                img2 = self.patch_bank[label_pair[1]][rand2][j]
                j+=1

            # concatenate these images up and down
            img = np.concatenate((img1, img2), axis=0)
            img = img_to_tensor_format(img, new_size=(IMG_WIDTH, IMG_WIDTH * 2), to_rgb=True)
            arr[i*3:i*3+3] = img
        return arr

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        pref_idx = idx % self.num_preferences  # which preference
        idx = idx % self.num_preferences      # which example in preference pref_idx
        example = self.data_map[pref_idx][idx]  
        # randomly reorder the tuples in example
        random.shuffle(example)
        arr = self.get_sample(example)
        return arr, pref_idx, example

    

class ContextDataModule(pl.LightningDataModule):
    def __init__(self, pref_config, context_data_config_path, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.pref_config = pref_config
        self.context_data_config_path = context_data_config_path
        self.setup()

    def setup(self, stage=None):
        self.train_dataset = ContextDataset(True, self.pref_config, self.context_data_config_path)
        self.val_dataset = ContextDataset(False, self.pref_config, self.context_data_config_path)
        self.test_dataset = ContextDataset(False, self.pref_config, self.context_data_config_path)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--paths_file', type=str, default='paths.yaml', help='Path to the paths config')
    args = parser.parse_args()
    with open(args.paths_file, 'r') as file:
        paths_config = yaml.safe_load(file)

    pref_config = os.path.join(project_dir, paths_config['Paths']['pref_config'])
    context_data_config_path = os.path.join(project_dir, paths_config['Paths']['context_data_config_path'])

    dataloader = DataLoader(ContextDataset(True, pref_config, context_data_config_path), batch_size = 12)
    batch = next(iter(dataloader))
    contexts = batch[0]
    viz_contexts(contexts)