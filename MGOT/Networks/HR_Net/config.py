import os
from easydict import EasyDict as edict
import time
import torch

# init
__C = edict()
cfg = __C

#------------------------------TRAIN------------------------
__C.SEED = 1  # random seed,  for reproduction


__C.NET = 'HR_Net'
__C.PRE_HR_WEIGHTS = '/HOME/scw6cs4/run/FIDT_main/Networks/HR_Net/hrnetv2_w48_imagenet_pretrained.pth'
