import torch
from torch import nn
import numpy as np

from model_factory.modules import *

class ChemTransformer(nn.Module):
  def __init__(self):
    super().__init__()
    self.d_input = 31
    self.d_model = 16
    self.n_cxt = 97
    self.h = 4
    self.L = 1

    # input shape (n_cxt, d_input)
    # output shape (n_cxt, d_model)
    self.transformer = Transformer(self.d_input, self.d_model, self.n_cxt, self.h, self.L)

    # input shape (n_cxt, d_model)
    # output shape (n_cxt, 1)
    self.project = nn.Linear(self.d_model, 1)

  # u : input shape (1, n, d_input)
  def forward(self, x):
    x = x.view(-1, self.d_input)
    output = self.transformer(x)
    output = self.project(output)
    return torch.mean(output)
    