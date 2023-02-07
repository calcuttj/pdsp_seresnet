import torch
import time
import process_hits
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from collections import OrderedDict as ODict
device = 'cpu'

"""
Utility function for computing output of convolutions
takes a tuple of (h,w) and returns a tuple of (h,w)
"""
def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w

class Stage1Block(nn.Module):
  def __init__(self):
    super().__init__()

    self.blocks = [
      ##k = 7, s = 2, p = 0
      nn.Conv2d(1, 64, 7, 2, bias=False),
      ##k = 3, s = 2, p = 0
      nn.MaxPool2d(3, stride=2)
    ]
    self.layers = nn.Sequential(*self.blocks)

  def forward(self, x):
    print('\tStage1')
    #return self.layers(x)
    for b in self.blocks:
      x = b(x)
    return x

class NBlock(nn.Module):
  def __init__(self, depth, insize, infilters, outfilters, ksize=3):
    super().__init__()

    deeper_size = (
      int((insize[0] + 2 - (ksize))/2 + 1),
      int((insize[1] + 2 - (ksize))/2 + 1)
    )

    self.blocks = [
      Stage2NBlock((insize if (i == 0 or outfilters == infilters) else deeper_size), (infilters if i == 0 else outfilters), outfilters=outfilters, ksize=ksize)
      for i in range(depth)
    ]

    self.layers = nn.Sequential(*self.blocks)

  def forward(self, x):
    return self.layers(x)

class SEBlock(nn.Module):
  def __init__(self, insize, nfilters, r=16):
    self.nfilters=nfilters
    super().__init__()
    self.layers = nn.Sequential(
      nn.AvgPool2d(insize),
      nn.Flatten(),
      nn.Linear(nfilters, nfilters // r),
      nn.ReLU(),
      nn.Linear(nfilters // r, nfilters),
      nn.Sigmoid(),
    )

  def forward(self, x):
    return self.layers(x).reshape(x.shape[0], self.nfilters,1,1) * x

class Stage2NBlock(nn.Module):
  def __init__(self, insize,  infilters, outfilters=64, ksize=3, r=16):
    super().__init__()

    #Check if infilters < outfilters
    #create a 1x1 conv
    if outfilters > infilters:
      stride = 2
      self.increased_filters = True
      padding=1
      self.conv_1x1 = nn.Conv2d(infilters, outfilters, 1, 2, bias=False)
      self.insize = (
        int((insize[0] + 2*padding - (ksize))/2 + 1),
        int((insize[1] + 2*padding - (ksize))/2 + 1)
      )

    else:
      stride = 1
      self.increased_filters = False
      padding='same'
      self.insize = insize


    self.blocks = [
        nn.BatchNorm2d(infilters),
        nn.ReLU(),
        nn.Conv2d(infilters, outfilters, ksize, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(outfilters),
        nn.ReLU(),
        nn.Conv2d(outfilters, outfilters, ksize, stride=1, padding='same', bias=False),
        SEBlock(self.insize, outfilters, r),
    ]
    self.layers = nn.Sequential(*self.blocks)


  def forward(self, x):
    import os, psutil;
    process = psutil.Process(os.getpid())
    print('\tStage2N')
    print(process.memory_info().rss*1e-9)
    #y = self.layers(x)
    y = self.blocks[0](x)
    print(0, process.memory_info().rss*1e-9)
    for i,b in enumerate(self.blocks[1:]):
      y = b(y)
      print(i, process.memory_info().rss*1e-9)
    print(process.memory_info().rss*1e-9)

    if self.increased_filters:
      y = torch.add(self.conv_1x1(x), y)
    else:
      y = torch.add(x, y)
    print(process.memory_info().rss*1e-9)

    return y

class SEResnetPDSP(nn.Module):
  def __init__(self, depth=[4, 6, 3], nfilters=[225, 256, 512],
               ntimes=913, nwires=[800, 800, 480]):
    super().__init__()

    self.nwires = nwires
    #1 stage1 block for each
    self.stage1 = [Stage1Block() for i in nwires]

    self.stage1_shapes = [
      conv_output_shape(
        conv_output_shape((ntimes, nw), kernel_size=7, stride=2),
        kernel_size=3, stride=2
      ) for nw in nwires
    ]

    #3 stage 2 blocks for each branch
    #do not increase filters here
    self.stage2 = [
      #NBlock(3, (ntimes, nwires[i]), 64, 64) for i in range(len(nwires))
      NBlock(3, self.stage1_shapes[i], 64, 64) for i in range(len(nwires))
    ]

    self.stage1_2_layers = [
      #nn.Sequential(s1) for s1, s2 in zip(self.stage1, self.stage2)
      nn.Sequential(s1, s2) for s1, s2 in zip(self.stage1, self.stage2)
    ]

    ##use plane 0 -- plane 2 will be padded to its size
    self.stage3n_shapes = [self.stage1_shapes[0]]
    for i in range(1, len(depth)):
      self.stage3n_shapes.append(conv_output_shape(self.stage3n_shapes[i-1], kernel_size=3, stride=2, pad=1))

    final_shape = conv_output_shape(self.stage3n_shapes[-1], kernel_size=3, stride=2, pad=1)

    print('Shapes', self.stage3n_shapes)
    full_nfilters = [64*3, *nfilters]
    print(full_nfilters)
    self.stage3n = [
      NBlock(depth[i], self.stage3n_shapes[i], full_nfilters[i], full_nfilters[i+1])
      for i in range(3) #range(len(depth))
    ]
    for i in range(len(depth)): print(full_nfilters[i], full_nfilters[i+1])
    self.stage3_layers = nn.Sequential(*self.stage3n)

    self.final_layers = nn.Sequential(
      #batch norm
      nn.BatchNorm2d(nfilters[-1]),
      #relu
      nn.ReLU(),
      #global avg pool
      nn.AvgPool2d(final_shape),
      #linear
      nn.Flatten(),
      nn.Linear(nfilters[-1] ,4),
      #softmax
      nn.Softmax()
    )

  def forward(self, x):

    p2_start_initial = (self.nwires[0] - self.nwires[2]) // 2


    plane0_in = x[:, 0, :, :].reshape(x.shape[0], 1, *x.shape[2:])
    plane1_in = x[:, 1, :, :].reshape(x.shape[0], 1, *x.shape[2:])
    plane2_in = x[:, 2, :, p2_start_initial:-p2_start_initial]
    plane2_in = plane2_in.reshape(x.shape[0], 1, *plane2_in.shape[1:])
    planes = [plane0_in, plane1_in, plane2_in]
    print(plane0_in.shape)

    print('Stages 1 & 2')
    stage1_outs = []
    for layer,x in zip(self.stage1_2_layers, planes):
      stage1_outs.append(layer(x))

    print('Done')

    #y = self.stage1_2_layers[2](x)

    print('Padding')
    stage1_p2_pad = torch.zeros(x.shape[0], 64, *self.stage1_shapes[0])
    print(stage1_p2_pad.shape)
    p2_start = (self.stage1_shapes[0][1] - self.stage1_shapes[2][1])//2

    stage1_p2_pad[:, :, :, p2_start:-p2_start] = stage1_outs[2]
    stage1_outs = [stage1_outs[0], stage1_outs[1], stage1_p2_pad]
    y = torch.cat(stage1_outs, 1)

    print(y.shape)
    y = self.stage3_layers(y)

    y = self.final_layers(y)

    return y 

def load_data(files):
  pdsp_datas = [process_hits.PDSPData(f) for f in files]
  planes = [[] for i in range(3)]
  truths = [] 

  a = 0 
  for pd in pdsp_datas:
    print(a, end='\r')
    a += 1
    pd.load_truth()
    for i, t in zip(range(pd.nevents), pd.topos):
      if t < 0: continue

      truths.append(t)
      pd.get_event(i)

      for j in range(3):
        planes[j].append(pd.make_plane(j, pad2=True))

  return (planes, truths)


def train(model, loss_fn, optimizer, pdsp_data):

  if not pdsp_data.loaded_truth: pdsp_data.load_truth()
  pdsp_data.clean_events()

  for batch, (X, y) in enumerate(
      pdsp_data.get_training_batches(batchsize=32)):
    #X, y = X.to(device), y.to(device)

    # Compute prediction error
    pred = model(torch.from_numpy(X).float())
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch % 100 == 0:
      loss, current = loss.item(), batch * len(X)
      print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

