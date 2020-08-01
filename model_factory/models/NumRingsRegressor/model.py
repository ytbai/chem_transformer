import torch
import numpy as np
from torch import nn
from model_factory.modules import *


class NumRingsRegressor(nn.Module):
  def __init__(self):
    super().__init__()
    self.d_model = 64
    self.n_cxt = 97
    self.num_classes = 9
    self.head = RegressorHead(self.d_model, self.n_cxt)

  def forward(self, x):
    output = self.head(x)
    if self.training:
      return output
    else:
      return torch.round(output)