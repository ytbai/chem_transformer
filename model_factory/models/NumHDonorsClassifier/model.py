import torch
import numpy as np
from torch import nn
from model_factory.modules import *

class NumHDonorsClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.d_model = 64
    self.n_cxt = 97
    self.num_classes = 12
    self.head = ClassifierHead(self.d_model, self.n_cxt, self.num_classes)

  def forward(self, x):
    output = self.head(x)
    if self.training:
      return output
    else:
      return torch.argmax(output, dim = 1, keepdim = False)