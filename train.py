from pdsp_data import PDSPDataset
import process_hits
import torch
from torch import nn
import numpy as np
from pdsp_resnet_pt import SEResnetPDSP_Plane2 as seresnet
from torch.utils.data import DataLoader
from argparse import ArgumentParser as ap

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

class MPOutputs:
  def __init__(self, world_size):
    self.world_size = world_size
    self.losses = []
    self.val_losses = []
    self.accuracies = []
    for i in range(world_size):
      self.losses.append([])
      self.val_losses.append([])
      self.accuracies.append([])

  def __print__(self):
    print(self.losses)
    print(self.val_losses)
    print(self.accuracies)


def get_device():
  return 'cuda' if torch.cuda.is_available() else 'cpu'

def get_splits(size, fracs=[.9, .1]):
  results = [int(fracs[0] * size)]
  results.append(size - results[0])
  return results

def make_loaders(dataset, batch_size=32, validate=False):

  if validate:
    from torch.utils.data import random_split
    print(len(dataset))
    splits = get_splits(len(dataset)) 
    print(splits)
    train_dataset, val_dataset = random_split(
        dataset, splits)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        sampler=DistributedSampler(train_dataset),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        sampler=DistributedSampler(val_dataset),
    )

  else:
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        sampler=DistributedSampler(dataset)
    )
    val_loader = None

  return (train_loader, val_loader)


def validate_loop(rank, loader, model, loss_fn, losses_list, acc_list,  max_iter=-1):
  device = get_device()
  size = len(loader.dataset)
  correct = 0
  
  losses_list.append([])
  with torch.no_grad():
    for batch, (x, y) in enumerate(loader):
      if max_iter > 0 and batch >= max_iter: break
      # Compute prediction error

      x = x.float().to(rank)
      y = y.long().argmax(1).to(rank)

      pred = model(x)
      loss = loss_fn(pred, y)
      loss, current = loss.item(), batch * len(x)
      losses_list[-1].append(loss)
      print(pred.argmax(1), y)
      correct += (pred.argmax(1) == y).type(torch.float).sum().item()
      print(f"loss: {loss:>7f}  [{current:>5d}/{size}]")

  correct /= size
  acc_list.append(correct)
  print(f'Validation accuracy: {100.*correct}')


def train_loop(rank, loader, model, loss_fn, optimizer, losses_list, lrs_list, scheduler=None, max_iter=-1):
  device = get_device()

  size = len(loader.dataset)

  losses_list.append([])
  lrs_list.append([])
  for batch, (x, y) in enumerate(loader):
    if max_iter > 0 and batch >= max_iter: break

    # Compute prediction error
    pred = model(x.float().to(rank))
    loss = loss_fn(pred, y.long().argmax(1).to(rank))

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Adjusting learning rate if available
    if scheduler:
      lrs_list[-1].append(scheduler.get_last_lr())
      scheduler.step()

    loss, current = loss.item(), batch * len(x)
    print(f"loss: {loss:>7f}  [{current:>5d}/{size}]")
    #print(pred, y)
    print(pred.argmax(1), y.argmax(1))
    losses_list[-1].append(loss)


def save_checkpoint(model, optimizer, scheduler, epoch):
  state = {
      'epoch':epoch,
      'model_state_dict':model.state_dict(),
      'optimizer_state_dict':optimizer.state_dict(),
    }
  if scheduler:
    state['scheduler_state_dict'] = scheduler.state_dict()

  torch.save(
    state,
    'checkpoint.pt'
  )

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def setup_trainers(filters, rank=0, weights=[], schedule=False, do_ddp=False):
  plane2_net = seresnet(nfilters=filters)
  if torch.cuda.is_available():
    print('Found cuda. Sending to gpu', rank)
    plane2_net.to(rank)

  if len(weights) == 0:
    print('Not weighting') 
    loss_fn = nn.CrossEntropyLoss()
  else:
    print('Weighting', weights)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights).float().to(rank))

  optimizer = torch.optim.SGD(plane2_net.parameters(), lr=0.0001, momentum=0.9)
  if do_ddp: plane2_net = DDP(plane2_net, device_ids=[rank])

  scheduler = (
    torch.optim.lr_scheduler.CyclicLR(optimizer,
                                      base_lr=0.0001,
                                      max_lr=0.001) if schedule
    else None)

  return (plane2_net, loss_fn, optimizer, scheduler)
  

def train(rank: int, filters, world_size: int, dataset, validate=False, batch_size=32, epochs=1, save=False, max_iter=-1, save_every=10, weights=[], schedule=False):

  ddp_setup(rank, world_size)

  #Get train and validate (if available) data loaders
  train_loader, val_loader = make_loaders(dataset, batch_size=batch_size, validate=validate)

  model, loss_fn, optimizer, scheduler = setup_trainers(filters, rank, weights, schedule, do_ddp=True)

  #Set up outputs
  losses = []
  lrs = []
  if validate:
    val_losses = []
    accuracies = []

  #Start epoch loop
  for e in range(epochs):
    print('Start epoch', e)

    train_loop(rank, train_loader, model, loss_fn, optimizer, losses, lrs, scheduler=scheduler, max_iter=max_iter)
    if e % save_every == 0 and rank == 0:
      save_checkpoint(model, optimizer, scheduler, e)

    if validate:
      print('Validating')
      validate_loop(rank, val_loader, model, loss_fn, val_losses, accuracies, max_iter=max_iter)
  
  if save and rank == 0:
    import h5py as h5 
    import time, calendar
    
    with h5.File(f'pdsp_training_losses_{calendar.timegm(time.gmtime())}.h5', 'a') as h5out:
      h5out.create_dataset('losses', data=np.array(losses))
      h5out.create_dataset('lrs', data=np.array(lrs))
      if validate:
        h5out.create_dataset('val_losses', data=np.array(val_losses))
        h5out.create_dataset('accuracies', data=np.array(accuracies))
  destroy_process_group()
    
if __name__ == '__main__':
  parser = ap()
  parser.add_argument('-f', required=True)
  parser.add_argument('--filters', nargs=3, default=[128, 192, 256], type=int)
  parser.add_argument('--batchsize', type=int, default=32)
  parser.add_argument('--epochs', type=int, default=1)
  parser.add_argument('--save', action='store_true')
  parser.add_argument('--validate', action='store_true')
  parser.add_argument('--weight', action='store_true')
  parser.add_argument('--cp_freq', type=int, default=10)
  parser.add_argument('--schedule', action='store_true')
  args = parser.parse_args()


  pdsp_data = process_hits.PDSPData(maxtime=500, linked=True)
  pdsp_data.load_h5(args.f)
  pdsp_data.clean_events()
  pdsp_dataset = PDSPDataset(pdsp_data)
  print(pdsp_dataset.pdsp_data.get_sample_weights())

  '''
  plane2_net = seresnet(nfilters=args.filters)
  if torch.cuda.is_available():
    print('Found cuda')
    plane2_net.to('cuda')'''

  world_size = torch.cuda.device_count() if torch.cuda.is_available else 1
  '''
  train(
      #plane2_net,
      0,
      args.filters,
      world_size,
      pdsp_dataset,
      validate=args.validate,
      weights=(pdsp_dataset.pdsp_data.get_sample_weights() if args.weight else []),
      batch_size=args.batchsize,
      epochs=args.epochs,
      save=args.save,
      save_every=args.cp_freq,
      schedule=args.schedule,
  )
  '''

  mp.spawn(train,
    args=(
      args.filters,
      world_size,
      pdsp_dataset,
      args.validate,
      args.batchsize,
      args.epochs,
      args.save,
      -1,
      args.cp_freq,
      (pdsp_dataset.pdsp_data.get_sample_weights() if args.weight else []),
      args.schedule,
    ), nprocs=world_size)

