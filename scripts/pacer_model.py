import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from context_encoder_model import VAE
from torch.nn import functional as F
from torchvision.utils import make_grid
import numpy as np
import os
import cv2

# Add the top-level project directory to the Python path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


""" Convolutional block:
    It follows a two 3x3 convolutional layer, each followed by a batch normalization and a relu activation.
"""
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

""" Encoder block:
    It consists of an conv_block followed by a max pooling.
    Here the number of filters doubles and the height and width half after every block.
"""
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

""" Decoder block:
    The decoder block begins with a transpose convolution, followed by a concatenation with the skip
    connection from the encoder block. Next comes the conv_block.
    Here the number filters decreases by half and the height and width doubles.
"""
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)

        return x


class UNET(pl.LightningModule):
    def __init__(self, paths_config = None):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)
        self.e5 = encoder_block(512, 1024)
        self.e6 = encoder_block(1024, 2048)

        """ Bottleneck """
        self.b = conv_block(2048, 4096)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.f1 = nn.Linear(4096 + 2048, 4096 + 1024)
        self.f2 = nn.Linear(4096 + 1024, 4096)

        """ Decoder """
        self.d0 = decoder_block(4096, 2048)
        self.d1 = decoder_block(2048, 1024)
        self.d2 = decoder_block(1024, 512)
        self.d3 = decoder_block(512, 256)
        self.d4 = decoder_block(256, 128)
        self.d5 = decoder_block(128, 64)

        """ Costmap output """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)


        if paths_config is not None:
            vae = VAE().load_from_checkpoint(os.path.join(project_dir, paths_config['Paths']['type1_vae32']))
            self.patch_encoder = vae.encoder
        else:
            self.patch_encoder = VAE().encoder

        mask_path = os.path.join(project_dir, "scripts/clean_mask.png")
        self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        self.mask = cv2.resize(self.mask, (256, 128))
        self.mask = np.array(self.mask)
        self.mask = self.mask / 255.0
        self.mask = np.expand_dims(self.mask, axis=0)
        self.mask = torch.tensor(self.mask).float()

        self.train_preds = None

    def forward(self, patches, bev_img):
        p = self.patch_encoder(patches)
        return self.forward_w_encoded_patches(p, bev_img)

    def forward_w_encoded_patches(self, p, bev_img):
        """ Encoder """
        s1, p1 = self.e1(bev_img)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        s5, p5 = self.e5(p4)
        s6, p6 = self.e6(p5)

        """ Bottleneck """
        b = self.b(p6)
        interpolate_size = b.shape[-2:]
        b = self.avgpool(b)
        b = torch.flatten(b, 1)
        x = torch.concat([b, p], dim=1)

        f1 = self.f1(x)
        f2 = self.f2(f1)
        f2 = f2.view(-1, 4096, 1, 1)
        f2 = F.interpolate(f2, size = interpolate_size)

        """ Decoder """
        d0 = self.d0(f2, s6)
        d1 = self.d1(d0, s5)
        d2 = self.d2(d1, s4)
        d3 = self.d3(d2, s3)
        d4 = self.d4(d3, s2)
        d5 = self.d5(d4, s1)

        """ Costmap output """
        outputs = self.outputs(d5)
        outputs = outputs * self.mask.to(outputs.device)

        return outputs
    
    def training_step(self, batch, batch_idx):
        patches = batch['context']
        bev_img = batch['bev_image']
        costmap = batch['costmap']

        bev_img = bev_img.float()
        patches = patches.float()
        costmap = costmap.float()

        # pass through the model
        outputs = self(patches, bev_img)

        # bce with logits loss with sum reduction
        loss = F.binary_cross_entropy_with_logits(outputs, costmap, reduction='mean')

        self.log('recon_loss', loss)

        self.train_preds = torch.sigmoid(outputs[:5])


        return loss
    
    def on_validation_epoch_end(self) -> None:
        bev_grid = make_grid(self.bevs, nrow=5).clone().detach()
        costmap_grid = make_grid(self.costmaps, normalize=False, nrow=5).clone().detach()
        pred_grid = make_grid(self.pred, normalize=False, nrow=5).clone().detach()

        if self.train_preds is not None:
            train_preds_grid = make_grid(self.train_preds, normalize=False, nrow=5).clone().detach()
            self.logger.experiment.add_image('train_preds', train_preds_grid, global_step=self.current_epoch)

        self.logger.experiment.add_image('bev', bev_grid, global_step=self.current_epoch)
        self.logger.experiment.add_image('costmap', costmap_grid, global_step=self.current_epoch)
        self.logger.experiment.add_image('output', pred_grid, global_step=self.current_epoch)

    def validation_step(self, batch, batch_idx):
        patches = batch['context']
        bev_img = batch['bev_image']
        costmap = batch['costmap']

        bev_img = bev_img.float()
        patches = patches.float()
        costmap = costmap.float()

        # pass through the model
        outputs = self(patches, bev_img)

        # bce with logits loss with sum reduction
        loss = F.binary_cross_entropy_with_logits(outputs, costmap, reduction='mean')

        self.log('val_recon_loss', loss)
        # get 5 random indices
        batch_size = bev_img.shape[0]
        indices = np.random.choice(batch_size, size=5, replace=False)
        self.bevs = bev_img[indices]
        self.costmaps = costmap[indices]
        self.pred = torch.sigmoid(outputs[indices])

        return loss
        
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-5) # maybe higher?
        return optimizer
    