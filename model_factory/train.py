import torch
import numpy as np

def train_once(model_factory, train_dataloader, criterion, optimizer):
  model_factory.model.train()
  loss_epoch = []
  for x, y_true, y_esol in train_dataloader:
    optimizer.zero_grad()
    y_pred = model_factory.model(x)
    loss = criterion(y_pred, y_true)
    loss.backward()
    optimizer.step()

    loss_epoch.append(loss.item())

  loss_epoch = np.array(loss_epoch).mean()
  model_factory.append_loss("loss_train", loss_epoch)

  return loss_epoch

def valid_once(model_factory, valid_dataloader, criterion):
  model_factory.model.eval()
  loss_epoch = []
  for x, y_true, y_esol in valid_dataloader:
    y_pred = model_factory.model(x)
    loss = criterion(y_pred, y_true)
    loss_epoch.append(loss.item())
  loss_epoch = np.array(loss_epoch).mean()
  
  model_factory.append_loss("loss_valid", loss_epoch)

  return loss_epoch
