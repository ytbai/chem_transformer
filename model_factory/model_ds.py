import torch
import numpy as np

from model_factory.model_factory import *
from model_factory.models.ChemTransformer.model import *
from model_factory.models.MolWeightRegressor.model import *
from model_factory.models.PolarAreaRegressor.model import *
from model_factory.models.NumHDonorsClassifier.model import *
from model_factory.models.NumRingsClassifier.model import *
from model_factory.models.NumRotBondsClassifier.model import *


class DS():
  regressors = ["mol_weight", "polar_area"]
  classifiers = ["num_H_donors", "num_rings", "num_rot_bonds"]

  model_map = {"mol_weight": MolWeightRegressor(),
               "polar_area": PolarAreaRegressor(),
               "num_H_donors": NumHDonorsClassifier(),
               "num_rings": NumRingsClassifier(),
               "num_rot_bonds": NumRotBondsClassifier(),
              }

  def __init__(self, target_name):
    self.target_name = target_name
    self.set_ds_type()
    self.set_model()
    
    self.set_criterion()
    self.set_metric()

  def set_ds_type(self):
    if self.target_name in self.regressors:
      self.ds_type = "reg"
    elif self.target_name in self.classifiers:
      self.ds_type = "class"


  def set_model(self):
    self.model = self.model_map[self.target_name].cuda()

    self.model_factory = ModelFactory(self.model)

  def set_criterion(self):
    if self.ds_type == "reg":
      self.criterion = nn.MSELoss()
    elif self.ds_type == "class":
      self.criterion = nn.CrossEntropyLoss()

  def set_metric(self):
    if self.ds_type == "reg":
      self.metric = nn.MSELoss()
    elif self.ds_type == "class":
      self.metric = "acc"

