import numpy as np
import h5py as h5
from tensorflow.python.ops.numpy_ops import np_config
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
np_config.enable_numpy_behavior()

class PDSPData:
  def __init__(self, filename):
    self.h5in = h5.File(filename)
    self.tp = np.dtype([('integral', 'f4'),
                        ('rms', 'f4'), ('time', 'f4'), ('wire', 'f4')])

    self.hit_events_0 = np.array(self.h5in['plane_0_hits']['event_id'])
    self.hit_events_1 = np.array(self.h5in['plane_1_hits']['event_id'])
    self.hit_events_2 = np.array(self.h5in['plane_2_hits']['event_id'])
    self.hit_events = [self.hit_events_0, self.hit_events_1, self.hit_events_2]
    self.nhits = np.array(self.h5in['events']['nhits'])
    self.events = np.array(self.h5in['events']['event_id'])

    self.get_event(0)
    #self.plane0_data = self.load_data(0)
    #self.plane1_data = self.load_data(1)
    #self.plane2_data = self.load_data(2)

  def load_data(self, pid, eventindex): 

    #n_hits = self.h5in[f'plane_{pid}_hits']['wire'].shape[0]

    print(pid)
    index = np.array([0])
    print(index)
    data = np.zeros(self.nhits[eventindex][pid], dtype=self.tp)
    #data = np.zeros(self.nhits[index[0]][pid], dtype=self.tp)
    indexes = [np.all(i) for i in self.hit_events[pid] == self.events[eventindex]]
    #indexes = [np.all(i) for i in self.hit_events[pid] == self.events[index[0]]]


    for n in self.tp.names:
      if n == 'event_id':
        data[n] = np.array(self.h5in[f'plane_{pid}_hits'][n])[indexes]
      else:
        data[n] = np.array(self.h5in[f'plane_{pid}_hits'][n])[indexes].reshape(self.nhits[eventindex][pid])
        #data[n] = np.array(self.h5in[f'plane_{pid}_hits'][n])[indexes].reshape(self.nhits[index[0]][pid])
    return data

  def get_event(self, eventindex):
    self.plane0_data = self.load_data(0, eventindex=eventindex)
    self.plane1_data = self.load_data(1, eventindex=eventindex)
    self.plane2_data = self.load_data(2, eventindex=eventindex)


  def make_plane(self, pid):
    plane = np.zeros((913, 480 if pid == 2 else 800))

    if pid not in [0, 1, 2]:
      ##TODO -- throw exception
      return 0
    elif pid == 0: data = self.plane0_data
    elif pid == 1: data = self.plane1_data
    elif pid == 2: data = self.plane2_data

    for d in data:
      #if d['event_id'] == self.events[eventindex]: continue
      w = int(d['wire'])
      t = 912 - int((d['time'] - 500)/6.025)
      i = d['integral']
      plane[t,w] += i
  
    return plane
