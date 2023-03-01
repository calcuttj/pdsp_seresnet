import numpy as np
from math import ceil, sqrt, exp, pi
import h5py as h5
#from tensorflow.python.ops.numpy_ops import np_config
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers
import matplotlib.pyplot as plt
#np_config.enable_numpy_behavior()

class PDSPData:
  def __init__(self, maxtime=913, linked=False):
    self.maxtime=maxtime
    self.linked = linked
    self.tp = np.dtype([('integral', 'f4'),
                        ('rms', 'f4'), ('time', 'f4'), ('wire', 'f4')])

    '''
    with h5.File(filename) as h5in:
    #h5in = h5.File(filename)
      self.tp = np.dtype([('integral', 'f4'),
                          ('rms', 'f4'), ('time', 'f4'), ('wire', 'f4')])

      self.loaded_truth = False
      self.data = dict()

      if not linked:
        self.hit_events_0 = np.array(h5in['plane_0_hits']['event_id'])
        print(len(self.hit_events_0))
        self.hit_events_1 = np.array(h5in['plane_1_hits']['event_id'])
        print(len(self.hit_events_1))
        self.hit_events_2 = np.array(h5in['plane_2_hits']['event_id'])
        print(len(self.hit_events_2))

        self.nhits = np.array(h5in['events']['nhits'])
        print(len(self.nhits))
        self.events = np.array(h5in['events']['event_id'])
        print(len(self.events))

        for n in self.tp.names:
          self.data[n] = []
          for pid in range(3):
            self.data[n].append(np.array(h5in[f'plane_{pid}_hits'][n]))


      else:
        self.keys = h5in.keys()
        self.event_keys = []
        nhits = []
        self.hit_events_0 = dict()
        self.hit_events_1 = dict()
        self.hit_events_2 = dict()

        self.k_nevents = dict()

        events = []
        for k in self.keys:
          self.hit_events_0[k] = np.array(h5in[f'{k}/plane_0_hits/event_id'])
          self.hit_events_1[k] = np.array(h5in[f'{k}/plane_1_hits/event_id'])
          self.hit_events_2[k] = np.array(h5in[f'{k}/plane_2_hits/event_id'])

          events += [i for i in np.array(h5in[f'{k}/events/event_id'])]
          n_k_events = len(np.array(h5in[f'{k}/events/event_id']))
          self.event_keys += [k]*n_k_events
          self.k_nevents[k] = n_k_events
          #print(f'Added {n_k_events} events from {k}')
          for n in self.tp.names:
            self.data[f'{k}/{n}'] = []
            for pid in range(3):
              self.data[f'{k}/{n}'].append(np.array(h5in[f'{k}/plane_{pid}_hits'][n]))



          nhits += [i for i in np.array(h5in[f'{k}/events/nhits'])]
        #print(self.event_keys)


        self.nhits = np.array(nhits)
        self.events = np.array(events)

      self.hit_events = [self.hit_events_0, self.hit_events_1, self.hit_events_2]
      self.nevents = len(self.events)

      print(f'Loading truth')
      self.load_truth(h5in)
      #print('getting event')
      #if self.nevents > 0:
      #  self.get_event(0)
      '''

  '''
  def load_external(self, keys, events, hit_events_0, hit_events_1, hit_events_2, ):
    self.keys = keys
    self.events = events
    self.hit_event_0 = hit_events_0
    self.hit_event_1 = hit_events_1
    self.hit_event_2 = hit_events_2

    self.hit_events = [self.hit_events_0, self.hit_events_1, self.hit_events_2]
    self.nevents = len(self.events)
    for k in self.keys:
      self.hit_events_0[k] = np.array(h5in[f'{k}/plane_0_hits/event_id'])
      self.hit_events_1[k] = np.array(h5in[f'{k}/plane_1_hits/event_id'])
      self.hit_events_2[k] = np.array(h5in[f'{k}/plane_2_hits/event_id'])

      events += [i for i in np.array(h5in[f'{k}/events/event_id'])]
      n_k_events = len(np.array(h5in[f'{k}/events/event_id']))
      self.event_keys += [k]*n_k_events
      self.k_nevents[k] = n_k_events
      #print(f'Added {n_k_events} events from {k}')
      for n in self.tp.names:
        self.data[f'{k}/{n}'] = []
        for pid in range(3):
          self.data[f'{k}/{n}'].append(np.array(h5in[f'{k}/plane_{pid}_hits'][n]))



      nhits += [i for i in np.array(h5in[f'{k}/events/nhits'])]
      '''
 

  def load_h5(self, filename):
    with h5.File(filename, 'r') as h5in:
    #h5in = h5.File(filename)

      self.loaded_truth = False
      self.data = dict()

      if not self.linked:
        self.hit_events_0 = np.array(h5in['plane_0_hits']['event_id'][:])
        print(len(self.hit_events_0))
        self.hit_events_1 = np.array(h5in['plane_1_hits']['event_id'][:])
        print(len(self.hit_events_1))
        self.hit_events_2 = np.array(h5in['plane_2_hits']['event_id'][:])
        print(len(self.hit_events_2))

        self.nhits = np.array(h5in['events']['nhits'][:])
        print(len(self.nhits))
        self.events = np.array(h5in['events']['event_id'][:])
        print(len(self.events))

        for n in self.tp.names:
          self.data[n] = []
          for pid in range(3):
            self.data[n].append(np.array(h5in[f'plane_{pid}_hits'][n][:]))


      else:
        self.keys = [k for k in h5in.keys()]
        self.event_keys = []
        nhits = []
        self.hit_events_0 = dict()
        self.hit_events_1 = dict()
        self.hit_events_2 = dict()

        self.k_nevents = dict()

        events = []
        for k in self.keys:
          self.hit_events_0[k] = np.array(h5in[f'{k}/plane_0_hits/event_id'][:])
          self.hit_events_1[k] = np.array(h5in[f'{k}/plane_1_hits/event_id'][:])
          self.hit_events_2[k] = np.array(h5in[f'{k}/plane_2_hits/event_id'][:])

          events += [i for i in np.array(h5in[f'{k}/events/event_id'][:])]
          n_k_events = len(np.array(h5in[f'{k}/events/event_id'][:]))
          self.event_keys += [k]*n_k_events
          self.k_nevents[k] = n_k_events
          #print(f'Added {n_k_events} events from {k}')
          for n in self.tp.names:
            self.data[f'{k}/{n}'] = []
            for pid in range(3):
              self.data[f'{k}/{n}'].append(np.array(h5in[f'{k}/plane_{pid}_hits'][n][:]))



          nhits += [i for i in np.array(h5in[f'{k}/events/nhits'][:])]
        #print(self.event_keys)


        self.nhits = np.array(nhits)
        self.events = np.array(events)

      self.hit_events = [self.hit_events_0, self.hit_events_1, self.hit_events_2]
      self.nevents = len(self.events)

      print(f'Loading truth')
      self.load_truth(h5in)
      self.clean_events()
      #print('getting event')
      #if self.nevents > 0:
      #  self.get_event(0)

  def get_indices(self, pdg):
    return [i for i in range(len(self.pdg)) if self.pdg[i][0] == pdg]

  def load_truth(self, h5in):
    if self.loaded_truth:
      print('Already loaded truth info')
      return
    self.loaded_truth = True
    if not self.linked:
      if 'truth' in h5in.keys():
        found_truth = True
        self.pdg = np.array(h5in['truth']['pdg'][:]) 
        self.interacted = np.array(h5in['truth']['interacted'][:])
        self.n_neutron = np.array(h5in['truth']['n_neutron'][:])
        self.n_proton = np.array(h5in['truth']['n_proton'][:])
        self.n_piplus = np.array(h5in['truth']['n_piplus'][:])
        self.n_piminus = np.array(h5in['truth']['n_piminus'][:])
        self.n_pi0 = np.array(h5in['truth']['n_pi0'][:])

        self.pdg = np.ndarray.flatten(self.pdg)
        self.interacted = np.ndarray.flatten(self.interacted)
        self.n_neutron = np.ndarray.flatten(self.n_neutron)
        self.n_proton = np.ndarray.flatten(self.n_proton)
        self.n_piplus = np.ndarray.flatten(self.n_piplus)
        self.n_piminus = np.ndarray.flatten(self.n_piminus)
        self.n_pi0 = np.ndarray.flatten(self.n_pi0)

    else:
      found_truth = False 
      pdg = [] 
      interacted = []
      n_neutron = []
      n_proton = []
      n_piplus = []
      n_piminus = []
      n_pi0 = []
      self.k_ntruths = dict()
      for k in self.keys:
        if 'truth' in h5in[f'{k}'].keys():
          found_truth = True

          sub_pdg = [i for i in np.array(h5in[f'{k}/truth/pdg'][:])]

          pdg += sub_pdg
          interacted += [i for i in np.array(h5in[f'{k}/truth/interacted'][:])]
          n_neutron += [i for i in np.array(h5in[f'{k}/truth/n_neutron'][:])]
          n_proton += [i for i in np.array(h5in[f'{k}/truth/n_proton'][:])]
          n_piplus += [i for i in np.array(h5in[f'{k}/truth/n_piplus'][:])]
          n_piminus += [i for i in np.array(h5in[f'{k}/truth/n_piminus'][:])]
          n_pi0 += [i for i in np.array(h5in[f'{k}/truth/n_pi0'][:])]

          #print(f'Added {len(sub_pdg)} truths from {k}')
          self.k_ntruths[k] = len(sub_pdg)

      if found_truth:
        self.pdg = np.array(pdg)
        self.interacted = np.array(interacted)
        self.n_proton = np.array(n_proton)
        self.n_neutron = np.array(n_neutron)
        self.n_piplus = np.array(n_piplus)
        self.n_piminus = np.array(n_piminus)
        self.n_pi0 = np.array(n_pi0)

        self.pdg = np.ndarray.flatten(self.pdg)
        self.interacted = np.ndarray.flatten(self.interacted)
        self.n_neutron = np.ndarray.flatten(self.n_neutron)
        self.n_proton = np.ndarray.flatten(self.n_proton)
        self.n_piplus = np.ndarray.flatten(self.n_piplus)
        self.n_piminus = np.ndarray.flatten(self.n_piminus)
        self.n_pi0 = np.ndarray.flatten(self.n_pi0)

    if found_truth: self.get_truth_topos()

  def get_truth_topos(self):
    topos = []
    for i in range(len(self.pdg)):
      if self.pdg[i] not in [211, -13]:
        topos.append(-1)
      elif not self.interacted[i]:
        topos.append(3)
      elif (self.n_piplus[i] == 0 and self.n_piminus[i] == 0 and
            self.n_pi0[i] == 0):
        topos.append(0)
      elif (self.n_piplus[i] == 0 and self.n_piminus[i] == 0 and
            self.n_pi0[i] == 1):
        topos.append(1)
      else:
        topos.append(2)
    self.topos = np.array(topos)

  def load_data(self, pid, eventindex): 

    data = np.zeros(self.nhits[eventindex][pid], dtype=self.tp)

    if self.linked:
      key = self.event_keys[eventindex]
      indices = [np.all(i) for i in self.hit_events[pid][key] == self.events[eventindex]]
    else:
      print('getting indices')
      indices = [np.all(i) for i in self.hit_events[pid] == self.events[eventindex]]
      print('done')

    key = '' if not self.linked else f'{key}/'

    #print('Loading data from link', key)


    for n in self.tp.names:
      if n == 'event_id':
        #data[n] = np.array(self.h5in[f'{key}plane_{pid}_hits'][n])[indices]
        data[n] = self.data[f'{key}{n}'][pid][indices]
      else:
        #data[n] = np.array(self.h5in[f'{key}plane_{pid}_hits'][n])[indices].reshape(self.nhits[eventindex][pid])
        data[n] = self.data[f'{key}{n}'][pid][indices].reshape(self.nhits[eventindex][pid])
    return data

  def get_event(self, eventindex):
    self.plane0_data = self.load_data(0, eventindex=eventindex)
    self.plane1_data = self.load_data(1, eventindex=eventindex)
    self.plane2_data = self.load_data(2, eventindex=eventindex)


  def make_plane(self, pid, pad2=False, use_width=False):
    plane = np.zeros((self.maxtime, 480 if (pid == 2 and not pad2) else 800))

    if pid not in [0, 1, 2]:
      ##TODO -- throw exception
      return 0
    elif pid == 0: data = self.plane0_data
    elif pid == 1: data = self.plane1_data
    elif pid == 2: data = self.plane2_data


    bin_width=6.025
    for d in data:
      w = int(d['wire'])
      if pid == 2:
        if w > 479: continue
        if pad2: w = w + (800 - 480)//2
      elif w > 799: continue
      t = 912 - int((d['time'] - 500)/bin_width)
      if t >= self.maxtime: continue
      i = d['integral']

      if not use_width:
        plane[t,w] += i
      else:
        rms = d['rms']
        #Get the number of bins away -- 5 sigma
        nbins = ceil(5*rms/bin_width)
        for b in range(t - nbins, t + nbins + 1):
          if b >= self.maxtime: continue
          plane[b,w] += i*(bin_width/(rms*sqrt(2.*pi)))*exp(-.5*((t - b)*bin_width/rms)**2)
  
    return plane

  def clean_events(self, check_nhits=True):
  ##Check this -- not working
    nohits = np.any(self.nhits == 0, axis=1)
    print(len(nohits))
    indices = np.where(
      ((self.pdg != 211) & (self.pdg != -13)) |
      (nohits if check_nhits else check_nhits)
    )

    self.pdg = np.delete(self.pdg, indices)
    self.topos = np.delete(self.topos, indices)
    self.interacted = np.delete(self.interacted, indices)
    self.n_neutron = np.delete(self.n_neutron, indices)
    self.n_proton = np.delete(self.n_proton, indices)
    self.n_piplus = np.delete(self.n_piplus, indices)
    self.n_piminus = np.delete(self.n_piminus, indices)
    self.n_pi0 = np.delete(self.n_pi0, indices)

    self.events = np.delete(self.events, indices, axis=0)
    self.nhits = np.delete(self.nhits, indices, axis=0)
    self.event_keys = np.delete(self.event_keys, indices)
    self.nevents -= len(indices[0])

  def get_nbatches(self, batchsize=2):
    return ceil(self.nevents/batchsize)

  def get_sample_weights(self):
    return 1./np.array([self.topos[self.topos == i].size for i in range(4)])

  def get_training_batches(self, batchsize=2, plane2_only=False, use_width=False, maxbatches=-1, startbatch=0):

    for i in range(startbatch, ceil(self.nevents/batchsize)):
      if maxbatches > 0 and i > maxbatches: break
      #print('Batch', i)
      #print('\t', np.arange(self.nevents)[i*batchsize:(i+1)*batchsize])

      batch_events = np.arange(self.nevents)[i*batchsize:(i+1)*batchsize]
      nb = len(batch_events)
      if plane2_only:
        plane_batch = np.zeros((nb, 1, self.maxtime, 480))
      else:
        plane_batch = np.zeros((nb, 3, self.maxtime, 800))

      truth_batch = np.zeros((nb, 4))
      for a, j in enumerate(batch_events):
        #print(j)
        self.get_event(j)
        if plane2_only:
          plane_batch[a,0,:,:] = self.make_plane(2, use_width=True)
        else:
          plane_batch[a,0,:,:] = self.make_plane(0, use_width=True)
          plane_batch[a,1,:,:] = self.make_plane(1, use_width=True)
          plane_batch[a,2,:,160:-160] = self.make_plane(2, use_width=True)

        truth_batch[a,self.topos[j]] = 1.

      yield (plane_batch, truth_batch)
