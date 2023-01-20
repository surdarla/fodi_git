import os
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW

from config import CFG
from utils import seed_everything
from prepare import prepare_imgs_and_targets

seed_everything(CFG.seed)
use_cuda = torch.cuda.is_available()
device = torch.cuda.device("cuda" if use_cuda else "cpu")

X,y = prepare_imgs_and_targets(CFG.data_dir, train=True)