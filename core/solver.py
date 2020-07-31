import os
from os.path import join as ospj
import time
import datetime
from munch import Munch

import torch
import torch.nn as nn
import torch.nn.functional as F

# import utils
from core.model import build_nets
from core.checkpoint import CheckpointIO
# from metrics.eval import calculate_metrics


class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.nets = build_nets(args)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            # utils.print_network(module, name)
            setattr(self, name, module)

        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)

            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), **self.nets),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_optims.ckpt'), **self.optims)]
        else:
            self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), **self.nets)]

        self.to(self.device)


    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def train(self, loaders):
        args = self.args
        nets = self.nets
        optims = self.optims

        # resume training if necessary
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        print('Start training...')
        start_time = time.time()
        for i in range(args.resume_iter, args.total_iters):
            train_loader = loaders.src
            for x in train_loader:
                # train the ResVAE
                resvae_loss, resvae_loss_ref = compute_ResVAE_loss(
                    nets, args, x)
                self._reset_grad()
                resvae.backward()
                optims.ResVAE.step()
            
            # save model checkpoints
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1)
            
            # print out log info
            if (i+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, args.total_iters)
                all_losses = dict()
                for loss, prefix in zip([resvae_loss_ref],
                                        ['ResVAE_loss',]):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                all_losses['G/lambda_ds'] = args.lambda_ds
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)


def compute_KL_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    loss = 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)
    return loss 

def compute_r_loss(x_real, x_recon):
    loss = F.mse_loss(x_real, x_recon)
    return loss

def compute_ResVAE_loss(nets, args, x):
    KL = compute_KL_loss(*nets.resvae.encoder(x))
    r_loss = compute_r_loss(x, nets.resvae.decoder(x))
    loss = KL + r_loss
    return loss, Munch(KL_loss=KL,
                       r_loss=r_loss,
                       reg=loss)


