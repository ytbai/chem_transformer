import torch
import numpy as np
from torch import nn

class Transformer(nn.Module):
  def __init__(self, d_input, d_model, n_cxt, h, L):
    super().__init__()
    self.d_input = d_input
    self.d_model = d_model
    self.n_cxt = n_cxt
    self.h = h
    self.L = L

    self.mod = nn.Sequential(
        TokenEmbed(self.d_input, self.d_model),
        PositionEmbed(self.d_model, self.n_cxt),
        EncoderStack(self.d_model, self.h, self.L)
    )
  
  # u : input can have shape (1, n, d_input) or (n, d_input)
  # output has shape (1, n, d_model)
  def forward(self, u):
    # shape (n, d_input)
    output = u.view(-1, self.d_input)
    output = self.mod(output)
    return output.view(1, -1, self.d_model)



class PositionEmbed(nn.Module):
  def __init__(self, d_model, n_cxt):
    super().__init__()

    self.d_model = d_model
    self.n_cxt = n_cxt

    # positional embedding matrix shape (n_cxt, d_model)
    self.position_mat = nn.Parameter(torch.randn(self.n_cxt, self.d_model))

  # x : shape (n, d_model)
  # output has same shape
  def forward(self, x):
    n = x.shape[0]
    return x + self.position_mat[:n]



class TokenEmbed(nn.Module):
  def __init__(self, d_input, d_model):
    super().__init__()
    self.d_input = d_input
    self.d_model = d_model

    # embedding matrix shape (d_input, d_model)
    self.embed_mat = nn.Parameter(torch.randn(self.d_input, self.d_model))

  # u : shape (n, d_input)
  # output shape (n, d_model)
  def forward(self, u):
    return torch.matmul(u, self.embed_mat)


class EncoderStack(nn.Module):
  def __init__(self, d_model, h, L):
    super().__init__()
    self.d_model = d_model
    self.h = h
    self.L = L

    self.encoder_layer_list = nn.ModuleList([EncoderLayer(self.d_model, self.h) for _ in range(self.L)])

  # x : shape (n, d_model)
  def forward(self, x):
    output = x
    for l in range(self.L):
      output = self.encoder_layer_list[l](x)

    return output

class EncoderLayer(nn.Module):
  def __init__(self, d_model, h):
    super().__init__()
    self.d_model = d_model
    self.h = h

    self.sa = SelfAttention(self.d_model, self.h)
    self.sa_ln = nn.LayerNorm(normalized_shape = (self.d_model, ))

    self.ff = PositionwiseFeedforward(self.d_model)
    self.ff_ln = nn.LayerNorm(normalized_shape = (self.d_model, ))

  def forward(self, x):
    sa_out = self.sa(x) + x
    sa_out = self.sa_ln(sa_out)

    ff_out = self.ff(sa_out)
    ff_out = self.ff_ln(ff_out)

    return ff_out

class PositionwiseFeedforward(nn.Module):
  def __init__(self, d_model, d_ff = None):
    super().__init__()
    self.d_model = d_model
    if d_ff is None:
      self.d_ff = 4*self.d_model
    else:
      self.d_ff = d_ff
    

    self.mod = nn.Sequential(
        nn.Linear(self.d_model, self.d_ff),
        nn.ReLU(),
        nn.Linear(self.d_ff, self.d_model)
    )
  
  # x : shape (n, d_model)
  # output has same shape
  def forward(self, x):
    return self.mod(x)

class SelfAttention(nn.Module):
  # d_model     : dimension of embedding vector
  # h   : number of heads in multi-head attention
  # Assume that d_model is divisible by num_heads
  def __init__(self, d_model, h):
    super().__init__()
    self.h = h
    self.d_model = d_model
    self.d_k = self.d_model // self.h # dimension of each query/key/value vector

    # every Q, K, V has shape (d_model, d_k)
    self.WQ_list = nn.ParameterList([nn.Parameter(torch.randn(self.d_model, self.d_k)) for _ in range(self.h)])
    self.WK_list = nn.ParameterList([nn.Parameter(torch.randn(self.d_model, self.d_k)) for _ in range(self.h)])
    self.WV_list = nn.ParameterList([nn.Parameter(torch.randn(self.d_model, self.d_k)) for _ in range(self.h)])

    # shape (d_model, d_model)
    self.WO = nn.Parameter(torch.rand(self.d_model, self.d_model))

  # x : shape (n, d_model)
  # output has same shape
  def forward(self, x):
    attention_list = []

    for k in range(self.h):
      # shape (n, d_k)
      query = torch.matmul(x, self.WQ_list[k])
      key = torch.matmul(x, self.WK_list[k])
      value = torch.matmul(x, self.WV_list[k])

      # shape (d_k, n)
      key_trans = key.permute(1, 0)

      # shape (n, n)
      query_key = torch.matmul(query, key_trans)
      query_key_norm = nn.functional.softmax(query_key / np.sqrt(self.d_k), dim=1)
      
      # shape (n, d_k)
      attention = torch.matmul(query_key_norm, value)

      attention_list.append(attention)
    
    multi_attention = torch.cat(attention_list, dim = 1)
    multi_attention = torch.matmul(multi_attention, self.WO)

    return multi_attention
