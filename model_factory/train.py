import torch
import numpy as np


def train(model_factory, train_dataset, criterion, optimizer):
  train_dataset.set_target("y_true")
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
  valid_dataset.set_target("y_true")
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
