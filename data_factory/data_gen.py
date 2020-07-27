import torch
import numpy as np
import pandas as pd
import os
from data_factory.make_embed import *

class DelaneyDataset(torch.utils.data.Dataset):
  def __init__(self, mode):
    self.mode           = mode
    self.set_file_address()
    self.df             = pd.read_csv(self.file_address)
    self.length         = self.df.shape[0]
    self.embed_obj      = Embed()
  
  def set_file_address(self):
    data_directory = "data_factory/data"
    if self.mode == "full":
      file_name = "delaney.csv"
    elif self.mode in ["train", "valid", "test"]:
      file_name = "delaney_"+self.mode+".csv"
    self.file_address = os.path.join(data_directory, file_name)

  def __getitem__(self, i):
    series              = self.df.iloc[i]
    compound_id         = series[0] 
    y_true              = series[1]
    y_esol              = series[2]
    smiles              = series[3].strip()
    x                   = self.embed_obj.embed_smiles(smiles)
    print(smiles)
    return x, y_true, y_esol
  
  def __len__(self):
    return self.length


def make_split():

  data_dir = "data_factory/data"
  file_name = "delaney.csv"
  file_address = os.path.join(data_dir, file_name)

  df = pd.read_csv(file_address)
  df = df.sample(frac=1, random_state=0).reset_index(drop=True)
  df_train = df.iloc[:700]
  df_valid = df.iloc[700:900]
  df_test = df.iloc[900:]

  df_train.to_csv(os.path.join(data_dir, "delaney_train.csv"), index=False)
  df_valid.to_csv(os.path.join(data_dir, "delaney_valid.csv"), index=False)
  df_test.to_csv(os.path.join(data_dir, "delaney_test.csv"), index=False)
