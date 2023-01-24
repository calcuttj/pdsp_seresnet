import process_hits
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
#import numpy as np


'''
class Blocks12Concat(layers.Layer):
  def __init__(self):
    self.branch0_block1 = Stage1Block()
    self.branch1_block1 = Stage1Block()
    self.branch2_block1 = Stage1Block()

    self.branch0_triple = TripleBlock()
    self.branch1_triple = TripleBlock()
    self.branch2_triple = TripleBlock()

  def call(self, inputs):
    x1 = self.branch0_block1(inputs)
'''

class NBlock(layers.Layer):
  def __init__(self, depth, nfilters, increased_filters=True):
    super().__init__()
    
    self.blocks = [
      Stage2NBlock(nfilters=nfilters,
                   increased_filters=(increased_filters if i == 0 else False))
      for i in range(depth)
    ]
    #self.block_1 = Stage2NBlock(increased_filters=increased_filters,
    #                            nfilters=nfilters)
    #self.block_2 = Stage2NBlock(nfilters=nfilters)
    #self.block_3 = Stage2NBlock(nfilters=nfilters)

  def call(self, inputs):
    x = self.blocks[0](inputs)
    for block in self.blocks[1:]:
      x = block(x)
      
    return x

class SEBlock(layers.Layer):
  def __init__(self, nfilters, r=16):
    super().__init__()
    self.global_avg_pool = layers.GlobalAveragePooling2D()
    self.nfilters = nfilters 
    self.fc1 = layers.Dense(self.nfilters // r, activation='relu')
    self.fc2 = layers.Dense(self.nfilters, activation='sigmoid')

  def call(self, inputs):
    x = self.global_avg_pool(inputs)
    x = layers.Reshape((1, 1, x.shape[-1]))(x)
    x = self.fc1(x)
    x = self.fc2(x)

    return layers.multiply([x, inputs])


class Stage1Block(layers.Layer):
  #def __init__(self, data):
  #def __init__(self, input_shape):
  def __init__(self):
    super().__init__()

    #print(data.shape[0], data.shape[1])
    self.conv2d = layers.Conv2D(
      #64, 7, input_shape=(*input_shape, 1), strides=(2, 2), use_bias=False,
      64, 7, strides=(2, 2), use_bias=False,
      padding='same', kernel_initializer='he_normal')

    self.maxpool = layers.MaxPooling2D((3,3), strides=(2,2), padding='same')

  def call(self, inputs):
    x = self.conv2d(inputs)
    return self.maxpool(x)

class Stage2NBlock(layers.Layer):
  #def __init__(self, input_shape, nfilters=64, ksize=3):
  def __init__(self, increased_filters=False, nfilters=64, ksize=3):

    super().__init__()

    #print(data.shape[-1])
    self.nfilters = nfilters
    self.increased_filters = increased_filters
    if self.increased_filters:
      self.conv_1x1 = layers.Conv2D(
          self.nfilters, 1, padding='same', strides=2,
          kernel_initializer='he_normal', use_bias=False)
      
    #if nfilters > input_shape[-1]:
    #  self.increased_filters = True
    #else: self.increased_filters = False

    self.batchnorm1 = layers.BatchNormalization(axis=-1)
    self.batchnorm2 = layers.BatchNormalization(axis=-1)
    self.relu = layers.ReLU()
    stride = 2 if increased_filters else 1

    self.conv1 = layers.Conv2D(nfilters, ksize, padding='same', strides=stride,
                               kernel_initializer='he_normal', use_bias=False)#,
                               #input_shape=input_shape)
    
    self.conv2 = layers.Conv2D(nfilters, ksize, padding='same', strides=1,
                               kernel_initializer='he_normal', use_bias=False)#,
                               #input_shape=input_shape)

    self.seblock = SEBlock(nfilters)

  def call(self, inputs):

    if self.increased_filters:
      '''
      xres = layers.Conv2D(
          self.nfilters, 1, padding='same', strides=2,
          kernel_initializer='he_normal', use_bias=False)(inputs)
      '''
      xres = self.conv_1x1(inputs)
                
    else:
      xres = inputs

    x = self.batchnorm1(inputs)
    x = self.relu(x)
    x = self.conv1(x)

    x = self.batchnorm2(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.seblock(x)
    return layers.add([x, xres])


class SEResNetPDSP(keras.Model):
  def __init__(self, depth=[4, 6, 3], nfilters=[128, 256, 512],
               ntimes=913, nwires=[800, 800, 480]):
    super().__init__()

    self.nwires = nwires

    self.pdsp_data = process_hits.PDSPData('pdsp_hits.h5')
    
    #self.br0_input = Input(shape=(ntimes, nwires[0], 1)) 
    #self.br1_input = Input(shape=(ntimes, nwires[1], 1)) 
    #self.br2_input = Input(shape=(ntimes, nwires[2], 1)) 

    #1 stage1 block for each
    self.stage1 = [Stage1Block() for i in nwires]

    #3 stage 2 blocks for each branch
    #do not increase filters here
    self.stage2 = [NBlock(3, 64, increased_filters=False) for i in nwires]

    self.stage3n = [
      NBlock(depth[i], nfilters[i], increased_filters=True)
      for i in range(len(depth)) 
    ]

    self.final_batchnorm = layers.BatchNormalization(axis=-1)
    self.final_relu = layers.Activation('relu')
    self.final_pool = layers.GlobalAveragePooling2D()
    self.final_dense = layers.Dense(4, activation='softmax')


  def call(self, data):

    #print(data)
    #self.pdsp_data.get_event(data)
    #plane_0 = self.pdsp_data.make_plane(0)
    #plane_1 = self.pdsp_data.make_plane(1)
    #plane_2 = self.pdsp_data.make_plane(2)

    #data will be of form (batch, time, wires, plane)
    #plane 2 will be padded equally on wire side to the same as planes 0/1
    #Trim these off. Later, the planes/branches will be concatenated together
    #and must be the same size. So we'll pad back later

    p2_start = (self.nwires[0] - self.nwires[2])//2
    print(p2_start)

    plane_0 = data[:,:,:,0]
    plane_1 = data[:,:,:,1]
    plane_2 = data[:,:,p2_start:-p2_start,2]
    planes = [plane_0, plane_1, plane_2]

    #call stage 1
    branch_outs = [
      #block(plane.reshape(1, *plane.shape, 1))
      block(plane.reshape(*plane.shape, 1))
      for block, plane in zip(self.stage1, planes)
    ]

    #call stage 2
    branch_outs = [
      block(out) for block, out in zip(self.stage2, branch_outs)
    ]
    for br in branch_outs:
      print(br.shape)

    #pad plane 2
    new_start = (branch_outs[0].shape[2] - branch_outs[2].shape[2]) // 2
    paddings = paddings = tf.constant([[0, 0], [0, 0], [new_start, new_start], [0, 0]]) 
    branch_outs[2] = tf.pad(branch_outs[2], paddings, 'CONSTANT')

    #concatenate
    x = layers.concatenate(branch_outs)

    #go through stages 3-N 
    for stage in self.stage3n:
      x = stage(x)

    #Add a final set of batch norm, relu activation, and pooling
    x = self.final_batchnorm(x)
    x = self.final_relu(x)
    x = self.final_pool(x)
    x = self.final_dense(x)

    return x


#def define_model():
