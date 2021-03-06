import torch
import numpy as np
import pandas as pd
import os
from data_factory.make_embed import *
from collections import Counter


class DelaneyDataset(torch.utils.data.Dataset):
  target_map = {"y_esol": 1,
                "min_deg": 2,
                "mol_weight": 3,
                "num_H_donors": 4,
                "num_rings": 5,
                "num_rot_bonds": 6,
                "polar_area": 7,
                "y_true": 8}

  def __init__(self, mode, target_name = "y_true"):
    self.mode           = mode
    self.set_file_address()
    self.df             = pd.read_csv(self.file_address)
    self.length         = self.df.shape[0]
    self.embed_obj      = Embed()
    self.esol    	= False
    self.set_target(target_name)

  def set_target(self, target_name):
    self.target_name = target_name
    self.target_index = self.target_map[self.target_name]
    return self

  def set_esol(self, arg = True):
    self.esol = arg
    return self

  def set_file_address(self):
    data_directory = "data_factory/data"
    file_name = "delaney_"+self.mode+".csv"
    self.file_address = os.path.join(data_directory, file_name)

  def get_target_mean(self):
    target_total = 0
    for i in range(self.length):
      x, target_value = self[i]
      target_total = target_total + target_value
    target_mean = target_total / self.length
    return target_mean.view((1,))

  def get_target_mode(self):
    counter = Counter()
    for i in range(self.length):
      x, target_value = self[i]
      counter[target_value.item()] += 1
    target_mode = torch.tensor(counter.most_common(1)[0][0]).type(torch.cuda.FloatTensor).view((1,))
    return target_mode

  def __getitem__(self, i):
    series              = self.df.iloc[i]
    
    if self.esol:
      y_true    	= torch.tensor(series[8]).type(torch.cuda.FloatTensor)
      y_esol		= torch.tensor(series[1]).type(torch.cuda.FloatTensor)
      return y_esol, y_true
    else:
      target_value      = torch.tensor(series[self.target_index]).type(torch.cuda.FloatTensor)

      smiles            = series[9].strip()
      x                 = self.embed_obj.embed_smiles(smiles).type(torch.cuda.FloatTensor)
      return x, target_value

  def __len__(self):
    return self.length


def make_split():
  data_dir = "data_factory/data"
  file_name = "delaney.csv"
  file_name_processed = "delaney-processed.csv"
  file_name_excluded = "excluded_molecules.csv"
  file_address_processed = os.path.join(data_dir, file_name_processed)
  file_address_excluded = os.path.join(data_dir, file_name_excluded)

  df_processed = pd.read_csv(file_address_processed)
  df_excluded = pd.read_csv(file_address_excluded)
  df = pd.concat([df_processed, df_excluded], ignore_index=True)

  df = df.sample(frac=1, random_state=0).reset_index(drop=True)
  df_train = df.iloc[:700]
  df_valid = df.iloc[700:900]
  df_test = df.iloc[900:]

  df.to_csv(os.path.join(data_dir, "delaney_full.csv"), index=False)
  df_train.to_csv(os.path.join(data_dir, "delaney_train.csv"), index=False)
  df_valid.to_csv(os.path.join(data_dir, "delaney_valid.csv"), index=False)
  df_test.to_csv(os.path.join(data_dir, "delaney_test.csv"), index=False)
