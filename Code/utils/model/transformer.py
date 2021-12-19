# https://huggingface.co/transformers/index.html

from transformers import ViTConfig, ViTModel, DeiTModel
import torch
from torch import nn

class ViT(nn.Module):
    def __init__(self, args):
        super(ViT, self).__init__()
        print("####this is Vit!####")

    def forward(self):
        print('forward')
