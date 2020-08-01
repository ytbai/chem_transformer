import torch
import numpy as np

from model_factory.model_factory import *
from model_factory.models.ChemTransformer.model import *
from model_factory.models.MolWeightRegressor.model import *
from model_factory.models.PolarAreaRegressor.model import *
from model_factory.models.NumHDonorsRegressor.model import *
from model_factory.models.NumRingsRegressor.model import *
from model_factory.models.NumRotBondsRegressor.model import *


class ModelDS():
  model_map = {"mol_weight": MolWeightRegressor(),
               "polar_area": PolarAreaRegressor(),
               "num_H_donors": NumHDonorsRegressor(),
               "num_rings": NumRingsRegressor(),
               "num_rot_bonds": NumRotBondsRegressor(),
              }

  def __init__(self, target_name):
    self.target_name = target_name
    self.set_model()

  def set_model(self):
    self.model = self.model_map[self.target_name].cuda()
    self.model_factory = ModelFactory(self.model)

