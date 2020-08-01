import torch
import numpy as np

def train_ds(ds, model_factory, train_dataset, optimizer):
  train_dataset.set_target(ds.target_name)
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 1, shuffle = True)
  ds.model.train()
  model_factory.model.eval()
  loss_epoch = []
  for x, target_true in train_dataloader:
    optimizer.zero_grad()
    features = model_factory.model.features(x).detach()
    target_pred = ds.model(features)
    loss = ds.criterion(target_pred, target_true)
    loss.backward()
    optimizer.step()

    loss_epoch.append(loss.item())

  loss_epoch = np.array(loss_epoch).mean()
  ds.model_factory.append_loss("loss_train", loss_epoch)

  return loss_epoch


def valid_ds(ds, model_factory, valid_dataset):
  valid_dataset.set_target(ds.target_name)
  valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size = 1, shuffle = False)
  ds.model.eval()
  model_factory.model.eval()
  loss_epoch = []
  for x, target_true in valid_dataloader:
    features = model_factory.model.features(x).detach()
    target_pred = ds.model(features)

    if isinstance(ds.metric, str) and ds.metric == "acc":
      loss = int((target_pred == target_true).item())
      loss_epoch.append(loss)
    else:
      loss = ds.metric(target_pred, target_true)
      loss_epoch.append(loss.item())
  loss_epoch = np.array(loss_epoch).mean()
  
  ds.model_factory.append_loss("loss_valid", loss_epoch)

  return loss_epoch
