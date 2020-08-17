import os
from os.path import join as ospj
import time
import datetime
from munch import Munch
from pathlib import Path

from PIL import Image
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F

# import utils
from core.model import build_nets, reparameterization
from core.checkpoint import CheckpointIO

class ModelUser(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.nets = build_nets(args)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            # utils.print_network(module, name)
            setattr(self, name, module)
        self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), **self.nets)]
        self._load_checkpoint(args.resume_iter)
        self.to(self.device)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def vectorization(self, imgPath):
        img = Image.open(Path(imgPath)).convert('RGB')
        print(self.nets.resvae.encoder)
        print(img)
        # vector = self.nets.resvae.encoder(img)
        # return vector
