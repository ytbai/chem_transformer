import torch
import numpy as np

def train_ds(model_ds, model_factory, train_dataset, criterion, optimizer):
  train_dataset.set_target(model_ds.target_name)
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 1, shuffle = True)
  model_ds.model.train()
  model_factory.model.eval()
  loss_epoch = []
  for x, target_true in train_dataloader:
    optimizer.zero_grad()
    features = model_factory.model.features(x).detach()
    target_pred = model_ds.model(features)
    loss = criterion(target_pred, target_true)
    loss.backward()
    optimizer.step()

    loss_epoch.append(loss.item())

  loss_epoch = np.array(loss_epoch).mean()
  model_ds.model_factory.append_loss("loss_train", loss_epoch)

  return loss_epoch


def valid_ds(model_ds, model_factory, valid_dataset, metric):
  valid_dataset.set_target(model_ds.target_name)
  valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size = 1, shuffle = False)
  model_ds.model.eval()
  model_factory.model.eval()
  loss_epoch = []
  for x, target_true in valid_dataloader:
    features = model_factory.model.features(x).detach()
    target_pred = model_ds.model(features)

    loss = metric(target_pred, target_true)
    loss_epoch.append(loss.item())
  loss_epoch = np.array(loss_epoch).mean()

  model_ds.model_factory.append_loss("loss_valid", loss_epoch)

  return loss_epoch
