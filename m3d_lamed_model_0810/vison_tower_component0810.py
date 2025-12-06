from __future__ import annotations
from typing import Union
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from .configuration_d3d_lamed import LamedConfig
from abc import ABC, abstractmethod
from torch import Tensor
import math
from typing import Any, Dict, List
import torch
import torch.nn as nn
from typing import Optional, Tuple, Type
from monai.networks.blocks import PatchEmbed
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from einops import rearrange
from einops.layers.torch import Rearrange
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.nets import ViT
from torchvision.models.video import swin3d_b

class Swin3D(nn.Module):

    def __init__(self):
        super(Swin3D, self).__init__()

        self.model = swin3d_b()
        self.model.head = nn.Identity()
        self.model.avgpool = nn.Identity()


    def forward(self, x):
        x = self.model(x)
        x = torch.reshape(x, (x.shape[0], self.hidden_size, -1))
        return x.permute(0, 2, 1)

    @property
    def hidden_size(self):
        return self.model.num_features


class Swin3DTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.select_layer = config.vision_select_layer
        self.select_feature = config.vision_select_feature

        self.vision_tower = Swin3D()

    def forward(self, images):
        last_feature = self.vision_tower(images)
        if self.select_layer == -1:
            image_features = last_feature
        else:
            raise ValueError(f'Unexpected select layer: {self.select_layer}')

        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def hidden_size(self):
        return self.vision_tower.hidden_size