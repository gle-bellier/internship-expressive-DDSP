import torch
from torch import nn
from diffusion_model import DBlock, UBlock

import sys

print(sys.version)

t = torch.randn(16, 2, 2000)

print("Down 1")
dblock = DBlock(2, 64)
t = dblock(t)

print("Down 2")
dblock = DBlock(64, 256)
t = dblock(t)

print("Up 1")
ublock = UBlock(256, 64)
t = ublock(t)
