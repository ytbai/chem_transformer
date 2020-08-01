import torch
import numpy as np

from model_factory.model_factory import *

def is_esol(model_factory):
  return isinstance(model_factory, str) and model_factory == "esol"

def test(model_factory, test_dataset, metric):
  if isinstance(model_factory, ModelFactory):
    model_factory.model.eval()
  test_dataset.set_esol(is_esol(model_factory))

  test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False)
  loss_epoch = []

  for x, y_true in test_dataloader:
    if is_esol(model_factory):
      y_pred = x
    elif isinstance(model_factory, torch.Tensor):
      y_pred = model_factory.view((1,))
    else:
      y_pred = model_factory.model(x)

    loss = metric(y_pred, y_true)
    loss_epoch.append(loss.item())

  loss_epoch = np.array(loss_epoch).mean()
  return loss_epoch
