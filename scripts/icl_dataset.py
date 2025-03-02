from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import importlib
import context_dataset
importlib.reload(context_dataset)
from context_dataset import ContextDataset
from torchvision.utils import make_grid
from argparse import ArgumentParser
from tqdm import tqdm
from utils import *
import pickle as pk
import numpy as np
import random
import yaml
import cv2
import os

def increase_contrast(costmap, mask):
    min_val = np.min(costmap[mask > 0])
    max_val = np.max(costmap[mask > 0])
    costmap = (costmap - min_val) / (max_val - min_val) 
    return costmap   

class ICLDataset(Dataset):
    def __init__(self, train, pref_config, data_config_path, augment = False, random_invert = False):
        
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
        self.context_data = ContextDataset(train=train, pref_config=pref_config, data_config_path=data_config_path)

        self.mask_image = cv2.imread("clean_mask.png", cv2.IMREAD_GRAYSCALE)
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
        skip = 0
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
            pref_costmaps = []
            filenames = [os.path.join(directory, f + ".pkl") for f in pickle_files]
            for f in tqdm(filenames, desc=f"Processing Costmaps in {directory}"):
                    with open (f, "rb") as f:
                        pref_costmaps += pk.load(f)
            self.costmaps.append(pref_costmaps)

    def load_extra_terrains(self): 
        ''' Load the synthetic terrains for data augmentation
        '''
        self.terrains_to_index = self.pref_config['preferences'][0]
        self.terrains = list(self.terrains_to_index.keys())

        extra_terrains_dir = self.pref_config['extra_terrains_dir']
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
        bev_image = self.bev_images[idx]

        # resize both
        bev_image = cv2.resize(bev_image, (256, 128))
        bev_costmap = cv2.resize(bev_costmap, (256, 128))
        bev_costmap = np.round(bev_costmap * 5) / 5
        og_image = bev_image.copy()

        # h, w = bev_image.shape
        # for r in range(h):
        #     for w in range(w):


        val_to_count = {}
        for i in range(0, 6):
            v = np.count_nonzero((bev_costmap == (i / 5.0)) & (mask > 0))
            if (v > 0):
                val_to_count[(i / 5.0)] = v

        # sort dict by value
        val_to_count = dict(sorted(val_to_count.items(), key=lambda item: item[1], reverse=True))
        index_to_terrain_img = {}
        for key in val_to_count:
            index_to_terrain_img[int(key*5)] = bev_image

        replaced_indices = set()
        l = len(index_to_terrain_img)//2

        # randomly choose synthetic terrains
        sample = random.sample(list(range(len(self.extra_terrains_images))), l)

        # iterate over the first half of val_to_count
        for key in list(val_to_count.keys())[:l]:
            replaced_indices.add(int(key*5))
            index_to_terrain_img[int(key*5)] = self.extra_terrains_images[sample.pop()]

        # replace terrains in the image
        for i in range(0, bev_image.shape[0]):
            for j in range(0, bev_image.shape[1]):
                bev_image[i][j] = index_to_terrain_img[int(bev_costmap[i][j] * 5)][i][j]

        # create a pref ordering from the non-zero keys
        pref_index_order = [int(i*5) for i in val_to_count.keys()]
        random.shuffle(pref_index_order)

        s = set()
        for i in val_to_count.keys():
            if val_to_count[i] > 2000 or len(s) < 3:
                s.add(int(i*5))
        
        p = []
        # make p a list of indices with counts greater than 2000 or length 3
        for i in pref_index_order:
            if i in s:
                p.append(i)

        pref_ordering = []
        for i in p:
            pref_ordering.append([self.terrains[i]])

        pref_combos = self.context_data.create_preference_combos(pref_ordering)
        pref = random.choice(pref_combos)
        context = self.context_data.get_sample(pref)

        # replace context with the synth data
        for i in range(3):
            for j in range(2):
                idx = self.terrains_to_index[pref[i][j]]
                if idx in replaced_indices:
                    # choose a random 64x64 patch from the synth image
                    synth_image = index_to_terrain_img[idx]
                    w = random.randint(0, synth_image.shape[1]-64)
                    h = random.randint(0, synth_image.shape[0]-64)
                    synth_patch = synth_image[h:h+64, w:w+64, :]
                    synth_patch = img_to_tensor_format(synth_patch, new_size=(32, 32), to_rgb=False, transpose=True, normalize = False)
                    # replace the corresponding patch in context with the synth data
                    context[3*i:3*(i+1), 32*j:32*(j+1), :] = synth_patch

        # prepare the costmap according to the new pref
        new_costmap = np.zeros(bev_costmap.shape)
        for i in range(0, bev_image.shape[0]):
            for j in range(0, bev_image.shape[1]):
                new_costmap[i][j] = pref_index_order.index(int(bev_costmap[i][j]*5))/5.0

        bev_image = np.transpose(bev_image, (2, 0, 1)) * self.mask_image
        new_costmap = np.expand_dims(new_costmap, axis=0)
        new_costmap = increase_contrast(new_costmap, self.mask_image)
        new_costmap = new_costmap * self.mask_image

        return {'context' : context.copy(),
                'bev_image' : bev_image.copy(),
                'bev_costmap' : bev_costmap.copy(),
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


class ICLDataModule(pl.LightningDataModule):
    def __init__(self, pref_config, data_config_path, batch_size=64, num_workers=4, augment = False, random_invert = False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = ICLDataset(train=True,
                                pref_config=pref_config,
                                data_config_path=data_config_path,
                                augment=augment,
                                random_invert=random_invert)
        self.val_dataset = ICLDataset(train=False,
                                pref_config=pref_config,
                                data_config_path=data_config_path,
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

    pref_config = paths_config['Paths']['pref_config']
    data_config_path = paths_config['Paths']['data_config_path']

    dataset = ICLDataset(train=True,
                          pref_config=pref_config,
                          data_config_path=data_config_path, augment=True)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    batch = next(iter(dataloader))
    
    viz_icl_dataset_batch(batch)


