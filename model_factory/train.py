import torch
import numpy as np

def train_ds(model_ds_factory, model_factory, train_dataset, criterion, optimizer):
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 1, shuffle = True)
  model_ds_factory.model.train()
  model_factory.model.eval()
  loss_epoch = []
  for x, target_true in train_dataloader:
    optimizer.zero_grad()
    features = model_factory.model.features(x).detach()
    target_pred = model_ds_factory.model(features)
    loss = criterion(target_pred, target_true)
    loss.backward()
    optimizer.step()

    loss_epoch.append(loss.item())

  loss_epoch = np.array(loss_epoch).mean()
  model_ds_factory.append_loss("loss_train", loss_epoch)

  return loss_epoch


def valid_ds(model_ds_factory, model_factory, valid_dataset, metric):
  valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size = 1, shuffle = False)
  model_ds_factory.model.eval()
  model_factory.model.eval()
  loss_epoch = []
  for x, target_true in valid_dataloader:
    features = model_factory.model.features(x).detach()
    target_pred = model_ds_factory.model(features)

    if isinstance(metric, str) and metric == "acc":
      loss = int((target_pred == target_true).item())
      loss_epoch.append(loss)
    else:
      loss = metric(target_pred, target_true)
      loss_epoch.append(loss.item())
  loss_epoch = np.array(loss_epoch).mean()
  
  model_ds_factory.append_loss("loss_valid", loss_epoch)

  return loss_epoch


def train(model_factory, train_dataset, criterion, optimizer):
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 1, shuffle = True)
  model_factory.model.train()
  loss_epoch = []
  for x, y_true in train_dataloader:
    optimizer.zero_grad()
    y_pred = model_factory.model(x)
    loss = criterion(y_pred, y_true)
    loss.backward()
    optimizer.step()

    loss_epoch.append(loss.item())

  loss_epoch = np.array(loss_epoch).mean()
  model_factory.append_loss("loss_train", loss_epoch)

  return loss_epoch

def valid(model_factory, valid_dataset, metric):
  valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size = 1, shuffle = False)
  model_factory.model.eval()
  loss_epoch = []
  for x, y_true in valid_dataloader:
    y_pred = model_factory.model(x)
    loss = metric(y_pred, y_true)
    loss_epoch.append(loss.item())
  loss_epoch = np.array(loss_epoch).mean()
  
  model_factory.append_loss("loss_valid", loss_epoch)

  return loss_epoch
