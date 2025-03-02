import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
import yaml
from icl_dataset import ICLDataModule
import os
from icl_model import UNET

def train(paths_config, random_invert, augment):
    pref_config = paths_config['Paths']['pref_config']
    data_config_path = paths_config['Paths']['data_config_path']

    dm = ICLDataModule(pref_config, data_config_path, num_workers=12,
                       augment=augment, random_invert=random_invert)
    model = UNET(paths_config)
    save_dir =  paths_config['Paths']['model_save_dir']
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
                        augment=augment, random_invert=random_invert)

    chkpt = torch.load(checkpoint_path)
    model = UNET(paths_config)
    model.load_state_dict(chkpt['state_dict'])

    save_dir =  paths_config['Paths']['model_save_dir']
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
    parser.add_argument('--stage', type=int, default=1, help='Training stage 1, 2, or 3')
    args = parser.parse_args()
    with open(args.paths_file, 'r') as file:
        paths_config = yaml.safe_load(file)

    if args.stage == 1:
        random_invert = False
        augment = False
    elif args.stage == 2:
        random_invert = True
        augment = False
    elif args.stage == 3:
        random_invert = False
        augment = True

    pref_config = paths_config['Paths']['pref_config']
    data_config_path = paths_config['Paths']['data_config_path']
    train(paths_config, random_invert, augment)