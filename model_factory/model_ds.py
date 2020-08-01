import torch
import numpy as np

def DSModel(target_name):
  if target_name == "polar_area":
    return PolarAreaRegressor()

def DSCriterion(target_name):
  if target_name == "polar_area":
    return nn.MSELoss()

def DSMetric(target_name):
  if target_name == "polar_area":
    return nn.MSELoss()