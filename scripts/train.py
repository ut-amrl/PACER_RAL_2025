import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from context_encoder_model import VAE
from torch.nn import functional as F
from argparse import ArgumentParser
import yaml
from icl_dataset import ICLDataset, ICLDataModule
# from paths import p
import time
from tqdm import tqdm
from torchvision.utils import make_grid
import numpy as np
import os
import cv2
from icl_model import UNET

def train(paths_config):
    pref_config = paths_config['Paths']['pref_config']
    data_config_path = paths_config['Paths']['data_config_path']

    dm = ICLDataModule(pref_config, data_config_path, num_workers=12, augment=False, random_invert=True)
    model = UNET(paths_config)
    save_dir = '/scratch/luisamao/arthur_logs/' # change
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=save_dir+"checkpoints",
        filename='icl-{epoch:02d}-{val_recon_loss:.2f}',
        monitor='val_recon_loss',
        mode='min',
        save_top_k=5
    )
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=100,
        callbacks=[checkpoint_callback],
        logger = pl.loggers.TensorBoardLogger(save_dir)
    )
    trainer.fit(model, dm)

def train_from_checkpoint(paths_config, checkpoint_path):
    pref_config = paths_config['Paths']['pref_config']
    data_config_path = paths_config['Paths']['data_config_path']
    dm = ICLDataModule(pref_config, data_config_path, num_workers=12, 
                       augment=False,
                       random_invert = True
                       )

    chkpt = torch.load(checkpoint_path)
    model = UNET(paths_config)
    model.load_state_dict(chkpt['state_dict'])

    save_dir = '/scratch/luisamao/arthur_logs/' # change
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=save_dir+"checkpoints",
        filename='icl-{epoch:02d}-{val_recon_loss:.2f}',
        monitor='val_recon_loss',
        mode='min',
        save_top_k=5
    )
    trainer = pl.Trainer(
        gpus=[1],
        max_epochs=100,
        callbacks=[checkpoint_callback],
        logger = pl.loggers.TensorBoardLogger(save_dir)
    )
    trainer.fit(model, dm)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--paths_file', type=str, default='paths.yaml', help='Path to the paths config')
    args = parser.parse_args()
    with open(args.paths_file, 'r') as file:
        paths_config = yaml.safe_load(file)

    pref_config = paths_config['Paths']['pref_config']
    data_config_path = paths_config['Paths']['data_config_path']
    print(pref_config, data_config_path)
    # exit()
    train(paths_config)
    # train_from_checkpoint("/scratch/luisamao/all_terrain/icl_vae_logs/icl_experiment2/checkpoints/icl-epoch=68-val_recon_loss=0.56.ckpt")
    exit()

    # pref_config_dir = p.pref_config_dir
    # data_config_path = p.data_config_path
    # pref_dir = p.pref_dir
    # bev_image_dir = p.bev_image_dir
    # dm = ICLDataModule(pref_config_dir, data_config_path, pref_dir, bev_image_dir, batch_size=128)
    # model = UNET()

    # train_loader = dm.train_dataloader()



    # # iterate through entire dataset
    # for inputs in tqdm(train_loader, desc='Iterating through dataset'):
    #     # print(inputs[0].shape, inputs[1].shape, inputs[2].shape)
    #     start = time.time()
    #     model.training_step(inputs, 0)
    #     end = time.time()
    #     print(f"Time taken: {end-start}")


    # exit()
    # inputs = next(iter(train_loader))
    # print(inputs[0].shape, inputs[1].shape, inputs[2].shape)
    # # exit()

    # outputs = model(inputs[0].float(), inputs[1].float())
    # print(outputs.shape)
    # # loss = model.training_step(inputs, 0)
    # # print(loss)

    # # save the output image
    # import cv2
    # output = outputs[0].detach().numpy()
    # # transpose
    # output = output.transpose(1, 2, 0)
    # # multiply by 255
    # output = output * 255
    # cv2.imwrite('output.png', output)

    # costmap = inputs[2][0].detach().numpy()
    # # multiply by 255
    # costmap = costmap * 255
    # costmap = costmap.transpose(1, 2, 0)
    # cv2.imwrite('costmap.png', costmap)

    # bev_image = inputs[1][0].detach().numpy()
    # # multiply by 255
    # bev_image = bev_image * 255
    # bev_image = bev_image.transpose(1, 2, 0)
    # # rgb to bgr
    # output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('bev_image.png', bev_image)

    # exit()




    # # input image is 749 x 1476 height width
    # # cut the height off at 576 + 64
    # # width goes to 1408+64

    # # create random 3 x 1472 x 640 tensor
    # # resize the image to 512 and 1024
    # # inputs = torch.randn((1, 3, 512, 1024))
    # inputs = torch.randn((1, 3, 128, 256))

    # # pass through the model
    # model = UNET()
    
    # outputs = model(inputs, torch.randn(1, 9, 32, 32))
    # # print(outputs.shape)

    # # print min and max of the output
    # print(outputs.min())
    # print(outputs.max())


