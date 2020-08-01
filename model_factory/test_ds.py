import torch
import numpy as np

def test_ds(model_ds, model_factory, test_dataset, metric):
  test_dataset.set_target(model_ds.target_name)
  test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False)
  model_ds.model.eval()
  model_factory.model.eval()
  loss_epoch = []
  for x, target_true in test_dataloader:
    features = model_factory.model.features(x).detach()
    target_pred = model_ds.model(features)

    loss = metric(target_pred, target_true)
    loss_epoch.append(loss.item())
  loss_epoch = np.array(loss_epoch).mean()

  return loss_epoch

def test_ds_mean(target_pred, target_name, test_dataset, metric):
  test_dataset.set_target(target_name)
  test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False)

  loss_epoch = []
  for x, target_true in test_dataloader:
    loss = metric(target_pred, target_true)
    loss_epoch.append(loss.item())
  loss_epoch = np.array(loss_epoch).mean()

  return loss_epoch