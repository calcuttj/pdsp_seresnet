import os
import h5py as h5
from torch.utils.data import Dataset
from process_hits import PDSPData
import numpy as np

class PDSPDataset(Dataset):
  def __init__(self, pdsp_data, plane_2=True, use_width=True):

    #print(f'Initializing PDSPData from {input_file}')
    self.pdsp_data = pdsp_data #PDSPData(maxtime=500, linked=True)
    #self.pdsp_data.load_h5(input_file)
    #print(f'Cleaning events')
    #self.pdsp_data.clean_events()

    #Set if we just want to look at plane_2
    self.plane_2 = plane_2
    self.use_width=use_width

  def __len__(self):
    return self.pdsp_data.nevents

  def __getitem__(self, idx):

    truth = np.zeros(4)
    truth[self.pdsp_data.topos[idx]] = 1.
    #if plane_2 only, just get the hit locations in plane_2
    self.pdsp_data.get_event(idx)
    if self.plane_2:
      data = self.pdsp_data.make_plane(2, use_width=self.use_width)
      return (data.reshape((1, *data.shape)), truth)
    else:
      data = np.zeros((3, self.pdsp_data.maxtime, 800))
      data[0, :, :] = self.pdsp_data.make_plane(0, use_width=self.use_width)
      data[1, :, :] = self.pdsp_data.make_plane(1, use_width=self.use_width)
      data[2, :, :] = self.pdsp_data.make_plane(2, use_width=self.use_width, pad2=True)
      return (data, truth)
