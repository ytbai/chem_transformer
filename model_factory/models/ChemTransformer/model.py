import torch
from torch import nn
import numpy as np

from model_factory.modules import *

class ChemTransformer(nn.Module):
  def __init__(self):
    super().__init__()
    self.d_input = 31
    self.d_model = 64
    self.n_cxt = 97
    self.h = 8
    self.L = 1

    # input shape (n_cxt, d_input)
    # output shape (n_cxt, d_model)
    self.transformer = Transformer(self.d_input, self.d_model, self.n_cxt, self.h, self.L)

    # input shape (n_cxt, d_model)
    # output shape (n_cxt, 1)
    self.project = nn.Linear(self.d_model, 1)

  # x : input shape (1, n, d_input)
  # output shape (1, d_model)
  def features(self, x):
    x = x.view(-1, self.d_input)
    output = self.transformer(x)
    output = torch.sum(output, dim = 0, keepdim = True)/self.n_cxt
    return output

  # x : input shape (1, n, d_input)
  # output shape (1, )
  def forward(self, x):
    output = self.features(x)
    output = self.project(output)
    return output.view((1,))