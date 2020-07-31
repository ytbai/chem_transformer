import torch
import numpy as np

# assume batch size 1
def test(model_factory, test_dataloader, criterion):
  model_factory.model.eval()
  loss_epoch = []
  for x, y_true in test_dataloader:
    y_pred = model_factory.model(x)
    loss = criterion(y_pred, y_true)
    loss_epoch.append(loss.item())
  loss_epoch = np.array(loss_epoch).mean()
  
  return loss_epoch
