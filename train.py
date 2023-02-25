from pdsp_data import PDSPDataset
import torch
from torch import nn
import numpy as np
from pdsp_resnet_pt import SEResnetPDSP_Plane2 as seresnet
from torch.utils.data import DataLoader
from argparse import ArgumentParser as ap

def get_device():
  return 'cuda' if torch.cuda.is_available() else 'cpu'

def make_loaders(dataset, batch_size=32, validate=False):
  if validate:
    from torch.utils.data import random_split
    train_dataset, val_dataset = random_split(
        dataset, [int(p * len(ds)) for p in [0.9, 0.1]])
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

  else:
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = None

  return (train_loader, val_loader)


def train_loop(loader, model, loss_fn, optimizer, losses_list, scheduler=None):
  device = get_device()

  size = len(loader.dataset)
  losses_list.append([])
  for batch, (x, y) in enumerate(loader):

    # Compute prediction error
    pred = model(x.float().to(device))
    loss = loss_fn(pred, y.float().to(device))

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Adjusting learning rate if available
    if scheduler: scheduler.step()

    loss, current = loss.item(), batch * len(x)
    print(f"loss: {loss:>7f}  [{current:>5d}/{size}]")
    losses_list[-1].append(loss)


def train(model, dataset, validate=False, batch_size=32, epochs=1, save=False):

  #Get train and validate (if available) data loaders
  train_loader, val_loader = make_loaders(dataset, validate=validate)

  #Make loss function, optimizer, and scheduler (if applicable)
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  scheduler= None #TODO

  #Set up outputs
  losses = []
  if validate: val_losses = []

  #Start epoch loop
  for e in range(epochs):
    print('Start epoch', e)

    train_loop(train_loader, model, loss_fn, optimizer, scheduler=scheduler)
    if validate: pass
  
  if save:
    import h5py as h5 
    import time, calendar
    
    with h5.File(f'pdsp_training_losses_{calendar.timegm(time.gmtime())}.h5', 'a') as h5out:
      h5out.create_dataset('losses', data=np.array(losses))
      if validate: pass
    
if __name__ == '__main__':
  parser = ap()
  parser.add_argument('-f', required=True)
  parser.add_argument('--filters', nargs=3, default=[128, 192, 256], type=int)
  parser.add_argument('--batchsize', type=int, default=32)
  parser.add_argument('--epochs', type=int, default=1)
  parser.add_argument('--save', action='store_true')
  parser.add_argument('--validate', action='store_true')
  args = parser.parse_args()

  pdsp_dataset = PDSPDataset(args.f)

  plane2_net = seresnet(nfilters=args.filters)
  if torch.cuda.is_available():
    print('Found cuda')
    plane2_net.to('cuda')

  train.train(
      plane2_net,
      pdsp_dataset,
      validate=args.validate,
      batch_size=args.batchsize,
      epochs=args.epochs,
      save=args.save,
  )

