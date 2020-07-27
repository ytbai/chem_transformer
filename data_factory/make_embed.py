import torch
import numpy as np
import pandas as pd
import os

class Embed():
  data_dir = "data_factory/data"
  embed_filename = "embed.csv"

  def __init__(self):
    self.get_embed()
    self.make_map()

  def get_embed(self):
    self.embed_df = pd.read_csv(os.path.join(self.data_dir, self.embed_filename))

  def make_map(self):
    self.map = dict()
    for index in range(len(self.embed_df)):
      char = self.embed_df.iloc[index]["char"]
      embed = self.embed_df.iloc[index]["embed"]
      self.map[char] = embed

    self.embed_dim = len(self.map)
  
  def embed_smiles(self, smiles):
    embed_index_list = torch.zeros(len(smiles)).type(torch.cuda.LongTensor)
    for i in range(len(smiles)):
      embed_index_list[i] = self.map[smiles[i]]
    
    return torch.nn.functional.one_hot(embed_index_list, num_classes = self.embed_dim)

  @classmethod
  def make_embed_csv(cls):
    data_file_address = os.path.join(cls.data_dir, "delaney.csv")
    data_df = pd.read_csv(data_file_address)

    char_set = set()
    for i in range(len(data_df)):
      for char in data_df.iloc[i]["SMILES"].strip():
        char_set.add(char)

    char_list = sorted(list(char_set))
    compress_df = pd.DataFrame({"embed": range(len(char_list)), "char": char_list})
    compress_df.to_csv(os.path.join(cls.data_dir, cls.embed_filename), index=False)
    