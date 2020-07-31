import torch
import numpy as np

def test(model_factory, test_dataset, metric):
  if isinstance(model_factory, str) and model_factory == "esol":
    return test_esol(test_dataset, metric)
  else:
    return test_model_factory(model_factory, test_dataset, metric)
    
def test_model_factory(model_factory, test_dataset, metric):
  model_factory.model.eval()
  test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False)
  loss_epoch = []
  for x, y_true in test_dataloader:
    y_pred = model_factory.model(x)
    loss = metric(y_pred, y_true)
    loss_epoch.append(loss.item())
  loss_epoch = np.array(loss_epoch).mean()

  return loss_epoch

def test_esol(test_dataset, metric):
  test_dataset.set_delaney()
  test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False)
  loss_epoch = []
  for y_true, y_esol in test_dataloader:
    loss = metric(y_esol, y_true)
    loss_epoch.append(loss.item())
  loss_epoch = np.array(loss_epoch).mean()
  
  test_dataset.set_delaney(False)
  return loss_epoch