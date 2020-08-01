import copy
import math

from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from core.wing import FAN

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.BatchNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

def reparameterization(latent_dim, mu, logvar, ):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), latent_dim))))
    z = sampled_z * std + mu
    return z

class Encoder(nn.Module):
    def __init__(self, img_size=512, max_conv_dim=512, target_size=8, latent_dim=50):
        super().__init__()
        self.latent_dim = latent_dim
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        ### down to target_size*2 img size by ResBlk 
        repeat_num = int(np.log2(img_size) - np.log2(target_size)) - 1
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 3, 2, 1)]
        blocks += [nn.BatchNorm2d(dim_out)]
        blocks += [nn.LeakyReLU(0.2)]
        self.main = nn.Sequential(*blocks)
        self.mu = nn.Linear(dim_out*target_size*target_size, latent_dim)
        self.logvar = nn.Linear(dim_out*target_size*target_size, latent_dim)
        print(self.main)


    def forward(self, img):
        print(img)
        out = self.main(img)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        mu = self.mu(out)
        logvar = self.logvar(out)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, img_size=512, start_size=8, latent_dim=50):
        super().__init__()
        self.latent_dim = latent_dim
        dim_in = 2**14 // img_size
        self.dim_in = dim_in
        self.start_size = start_size
        self.decoder_dense = nn.Sequential(
            nn.Linear(latent_dim, dim_in*start_size*start_size),
            nn.LeakyReLU(0.2)
        )
        repeat_num = int(np.log2(img_size) - np.log2(start_size)) + 1
        freeze_cnt = int(repeat_num/2)
        blocks = []
        for cnt in range(repeat_num-1):
            dim_out = dim_in if cnt <= freeze_cnt else int(dim_in/2)
            blocks += [nn.ConvTranspose2d(dim_in, dim_out, 3, 2, 1, 1)]
            blocks += [nn.LeakyReLU(0.2)]
            blocks += [nn.BatchNorm2d(dim_out)]
            dim_in = dim_out
        
        blocks += [nn.ConvTranspose2d(dim_in, 3, 2, 1, 1)]
        blocks += [nn.LeakyReLU(0.2)]
        
    def forward(self, z):
        batch_size = z.size(0)
        hidden = self.decoder_dense(z).view(
            batch_size, self.dim_in, self.start_size, self.start_size)
        recon_x = self.decoder_conv(hidden)
        return recon_x

class ResVAE(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.build_model(args)

    def build_model(self, args):
        self.encoder = Encoder(img_size=args.img_size,
                               target_size=args.target_size,
                               latent_dim=args.latent_dim)
        self.decoder = Decoder(img_size=args.img_size,
                               start_size=args.start_size,
                               latent_dim=args.latent_dim)

    def forward(self, x):
        z = reparameterization(latent_dim, *self.encoder(x))
        x_recon = self.decoder(z)
        return x_recon


def build_nets(args):
    resvae = ResVAE(args=args)
    nets = Munch(resvae=resvae,)
    return nets