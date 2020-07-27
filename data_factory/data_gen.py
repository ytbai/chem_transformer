import torch
import numpy as np
import pandas as pd
import os

class DelaneyDataset(torch.utils.data.Dataset):
  def __init__(self, mode):
    self.mode           = mode
    self.set_file_address()
    self.df             = pd.read_csv(self.file_address)
    self.length         = self.df.shape[0]
  
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
    smiles              = series[3]
    return smiles, y_true, y_esol
  
  def __len__(self):
    return self.length
