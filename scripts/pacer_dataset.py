from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import importlib
import context_dataset
importlib.reload(context_dataset)
from context_dataset import ContextDataset
from argparse import ArgumentParser
from tqdm import tqdm
from utils import *
import pickle as pk
import numpy as np
import random
import yaml
import cv2
import os

# Add the top-level project directory to the Python path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

IMG_WIDTH = 32

def increase_contrast(costmap, mask):
    min_val = np.min(costmap[mask > 0])
    max_val = np.max(costmap[mask > 0])
    costmap = (costmap - min_val) / (max_val - min_val) 
    return costmap   

class PACERDataset(Dataset):
    def __init__(self, train, pref_config, context_data_config_path, augment = False, random_invert = False):
        
        ''' Dataset for training PACER. Consists of a ContextDataset, BEV images, and target costmaps
                stage 1: no invert, no augment
                stage 2: with invert, no augment
                stage 3: no invert, with augment
        '''
        
        self.train = train
        self.random_invert = random_invert
        self.augment = augment

        self.costmaps = [] # 2d array. first dim is preference, second dim is the image
        self.bev_images = []
        self.random_invert = random_invert
        self.pref_config = yaml.load(open(pref_config, 'r'), Loader=yaml.FullLoader)
        self.load() 
        self.context_data = ContextDataset(train=train, pref_config=pref_config, context_data_config_path=context_data_config_path)

        mask_path = os.path.join(project_dir, "scripts/clean_mask.png")
        self.mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        self.mask_image = np.array(self.mask_image)
        self.mask_image = cv2.resize(self.mask_image, (256, 128))
        self.mask_image = np.expand_dims(self.mask_image, axis=0)
        self.mask_image = self.mask_image / 255.0

    def load(self):
        if self.train:
            pickle_files = self.pref_config["train"]
        else:
            pickle_files = self.pref_config["val"]

        filenames = [os.path.join(self.pref_config["bev_dir"], f+".pkl") for f in pickle_files]
        filenames = [os.path.join(self.pref_config['dataset_root_dir'], f) for f in filenames]
        for f in tqdm(filenames, desc="Processing Bev images"):
            with open (f, "rb") as f:
                self.bev_images += pk.load(f)

        for i in range(len(self.bev_images)):
            self.bev_images[i] = cv2.resize(self.bev_images[i], (256, 128))
       
        # if augment, only use the first pref_dir
        pref_dir = self.pref_config["costmap_dirs"]
        if self.augment:
            self.load_extra_terrains()
            pref_dir = [pref_dir[0]]
        # iterate through pref_dir
        for directory in pref_dir:
            directory = os.path.join(self.pref_config['dataset_root_dir'], directory)
            pref_costmaps = []
            filenames = [os.path.join(directory, f + ".pkl") for f in pickle_files]
            for f in tqdm(filenames, desc=f"Processing Costmaps in {directory}"):
                    with open (f, "rb") as f:
                        pref_costmaps += pk.load(f)
            for i in range(len(pref_costmaps)):
                pref_costmaps[i] = cv2.resize(pref_costmaps[i], (256, 128))
            self.costmaps.append(pref_costmaps)

    def load_extra_terrains(self): 
        ''' Load the synthetic terrains for data augmentation
        '''
        self.terrains_to_index = self.pref_config['preferences'][0]
        self.terrains = list(self.terrains_to_index.keys())

        extra_terrains_dir = os.path.join(self.pref_config['dataset_root_dir'], self.pref_config['extra_terrains_dir'])
        self.extra_terrains_images  = []
        for file in os.listdir(extra_terrains_dir):
            img = cv2.imread(os.path.join(extra_terrains_dir, file), cv2.IMREAD_COLOR)
            img = resize_image((128, 256), img)
            img = img_to_tensor_format(img, new_size=(256, 128), to_rgb=True, transpose=False)
            self.extra_terrains_images.append(img)    

    def __len__(self):
        if self.augment:
            return len(self.bev_images)
        return len(self.bev_images) * len(self.context_data)
    
    def get_randomly_inverted_sample(self, idx):
        sample = self.get_regular_sample(idx)

        context = sample['context']
        bev_costmap = sample['bev_costmap']
        bev_image = sample['bev_image']
        pref = sample['pref']

        # if idx is even, invert the costmap
        if idx % 2 == 0:
            sample['bev_costmap'] = 1 - bev_costmap
            # flip the context vertically
            # context has shape 9 x 64 x 32
            new_context = np.zeros(context.shape)
            new_context[:,0:32,:] = context[:,32:64,:]
            new_context[:,32:64,:] = context[:,0:32,:]
            sample['context'] = new_context
            return sample
        return sample
    
    def get_augmented_sample(self, idx):
        mask = self.mask_image[0]
        bev_costmap = self.costmaps[0][idx]
        bev_costmap = np.round(bev_costmap * 10) / 10
        bev_image = self.bev_images[idx]

        new_costmap = bev_costmap.copy()
        bev_image = bev_image.copy()

        h, w, c = bev_image.shape
        unique_costs = {}
        for r in range(h):
            for c in range(w):
                if mask[r][c]:
                    cost = bev_costmap[r][c]
                    unique_costs[cost] = unique_costs.get(cost, 0) + 1
        unique_costs = list(sorted(unique_costs.items(), key=lambda item: item[1], reverse=True))[:3]
        ordered_costs = sorted(cost for cost, count in unique_costs)
        synth_terrains = random.sample(self.extra_terrains_images, 3)
        for i in range(3):
            bev_image[bev_costmap == unique_costs[i][0]] = synth_terrains[i][bev_costmap == unique_costs[i][0]]

        # create a pref ordering from the non-zero keys
        pref_index_order = [0,1,2]
        random.shuffle(pref_index_order)

        pref =  [(pref_index_order[0], pref_index_order[1]),
                (pref_index_order[0], pref_index_order[2]),
                (pref_index_order[1], pref_index_order[2])] 
        random.shuffle(pref)

        # make context with the synth data
        context = np.empty((IMG_WIDTH * 2, IMG_WIDTH, 9))

        for i in range(3):
            context[:IMG_WIDTH, :, i*3: (i+1)*3] = synth_terrains[pref[i][0]][:IMG_WIDTH, :IMG_WIDTH, :]
            context[IMG_WIDTH:, :, i*3: (i+1)*3] = synth_terrains[pref[i][1]][:IMG_WIDTH, :IMG_WIDTH, :]
        context = np.transpose(context, (2, 0, 1))

        # prepare the costmap according to the new pref
        for i in range(3):
            new_costmap[bev_costmap == unique_costs[pref_index_order[i]][0]] = ordered_costs[i]

        bev_image = np.transpose(bev_image, (2, 0, 1)) * self.mask_image
        new_costmap = np.expand_dims(new_costmap, axis=0)
        new_costmap = increase_contrast(new_costmap, self.mask_image)
        new_costmap = new_costmap * self.mask_image

        return {'context' : context.copy(),
                'bev_image' : bev_image,
                'bev_costmap' : new_costmap,
                'pref' : pref.copy()}

    def get_regular_sample(self, idx):
        # print("regular")
        context, i1, pref = self.context_data[idx % len(self.context_data)]
        bev_costmap = self.costmaps[i1][idx // len(self.context_data)]
        bev_image = self.bev_images[idx // len(self.context_data)]
        bev_image = cv2.resize(bev_image, (256, 128))
        bev_image = np.transpose(bev_image, (2, 0, 1))
        bev_costmap = cv2.resize(bev_costmap, (256, 128))
        bev_costmap = np.expand_dims(bev_costmap, axis=0)
        bev_costmap = increase_contrast(bev_costmap, self.mask_image)

        bev_costmap = bev_costmap * self.mask_image
        return {'context' : context.copy(),
                'bev_image' : bev_image.copy(),
                'bev_costmap' : bev_costmap.copy(),
                'pref' : pref.copy()}

    def __getitem__(self, idx):
        if self.random_invert:
            return self.get_randomly_inverted_sample(idx)
        elif self.augment:
            return self.get_augmented_sample(idx)
        return self.get_regular_sample(idx)


class PACERDataModule(pl.LightningDataModule):
    def __init__(self, pref_config, context_data_config_path, batch_size=64, num_workers=4, augment = False, random_invert = False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = PACERDataset(train=True,
                                pref_config=pref_config,
                                context_data_config_path=context_data_config_path,
                                augment=augment,
                                random_invert=random_invert)
        self.val_dataset = PACERDataset(train=False,
                                pref_config=pref_config,
                                context_data_config_path=context_data_config_path,
                                augment=augment,
                                random_invert=random_invert)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    
# main method
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--paths_file', type=str, default='paths.yaml', help='Path to the paths config')
    args = parser.parse_args()
    with open(args.paths_file, 'r') as file:
        paths_config = yaml.safe_load(file)

    pref_config = os.path.join(project_dir, paths_config['Paths']['pref_config'])
    context_data_config_path = os.path.join(project_dir, paths_config['Paths']['context_data_config_path'])

    dataset = PACERDataset(train=True,
                          pref_config=pref_config,
                          context_data_config_path=context_data_config_path, augment=False)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    batch = next(iter(dataloader))
    
    viz_pacer_dataset_batch(batch)


