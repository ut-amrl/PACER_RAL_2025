import pytorch_lightning as pl
from torch import nn
import torch
from pl_bolts.models.autoencoders.components import (
    resnet50_decoder,
    resnet50_encoder,
)
from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule
from image_plotting_callback import ImageSampler
from argparse import ArgumentParser
from context_dataset import ContextDataModule

# source: https://colab.research.google.com/drive/1_yGmk8ahWhDs23U4mpplBFa-39fsEJoT?usp=sharing
class VAE(pl.LightningModule):
    def __init__(self, enc_out_dim=2048, latent_dim=1024, input_size=[2*32, 32]):
        super().__init__()

        self.save_hyperparameters()

        # encoder, decoder
        self.encoder = resnet50_encoder( False, False
        )
        self.encoder.conv1 = nn.Conv2d(9, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        self.decoder = resnet50_decoder(
            latent_dim=latent_dim,
            input_size=input_size,
            first_conv=False,
            maxpool1=False
        )

        self.decoder.conv1 = nn.Conv2d(256, 9, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl
    
    def forward(self, x):
        x_encoded = self.encoder(x)
        mu = self.fc_mu(x_encoded)
        x_hat = self.decoder(mu)

        scale = torch.exp(self.log_scale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # sample an image from the dist
        x_sample = dist.sample()

        return x_hat, x_sample

    def training_step(self, batch, batch_idx):
        x, _, _ = batch

        x = x.float()

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = (kl - recon_loss)
        elbo = elbo.mean()

        self.log_dict({
            'elbo': elbo,
            'kl': kl.mean(),
            'recon_loss': recon_loss.mean(),
            'reconstruction': recon_loss.mean(),
            'kl': kl.mean(),
        })

        return elbo


def train():
    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=None)
    parser.add_argument('--dataset', type=str, default='classifier')
    args = parser.parse_args()

    if args.dataset == 'cifar10':
        dataset = CIFAR10DataModule('.')
    if args.dataset == 'imagenet':
        dataset = ImagenetDataModule('.')
    if args.dataset == 'classifier':
        data_config_path = '/home/luisamao/sterling/spot_data/classifier_configs/all_data_config.yaml'
        pref_config_dir = "/home/luisamao/sterling/spot_data/preference_orderings/"
        dataset = ContextDataModule(pref_config_dir, data_config_path)

    sampler = ImageSampler()

    vae = VAE()
    save_dir = '/scratch/luisamao/all_terrain/icl_vae_logs/vae_logs3/'

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='elbo',
        dirpath=save_dir,
        filename='vae-{epoch:02d}-{recon_loss:.2f}',
        save_top_k=10,
        mode='min'
    )
    logger = pl.loggers.TensorBoardLogger(save_dir)
    trainer = pl.Trainer(
        gpus=[7,8],
        max_epochs=500,
        callbacks=[sampler, checkpoint_callback],
        logger = logger
    )
    trainer.fit(vae, dataset)


if __name__ == '__main__':
    train()
