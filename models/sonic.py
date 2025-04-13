import torch
from torch import nn

class Sonic(nn.Module):
    def __init__(self, args):
        self.vggt = VGGT(args)
        self.dp3 = dp3(args)


    def forward(self, images):
        output = self.vggt(images)
        action = self.dp3(images)
        return action

