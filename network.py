import tensorflow as tf
import tensorflow.contrib.layers as layers

def c2(inflow, filters, name):
  with tf.variable_scope(name):
    l1 = layers.conv2d(inflow, filters, 3)
    l2 = layers.conv2d(l1, filters, 3)
  return l2

def c4(inflow, filters, name):
  f1, f2, f3, f4 = filters
  with tf.variable_scope(name):
    l1 = layers.conv2d(inflow, f1, 3)
    l2 = layers.conv2d(l1, f2, 3)
    l3 = layers.conv2d(l2, f3, 3)
    l4 = layers.conv2d(l3, f4, 3)
  return l4

def c4c(inflow, filters, name):
  f1, f2, f3, f4, f5 = filters
  with tf.variable_scope(name):
    l1 = layers.conv2d(inflow, f1, 3)
    l2 = layers.conv2d(l1, f2, 3)
    l3 = layers.conv2d(l2, f3, 3)
    l4 = layers.conv2d(l3, f4, 3)
    l5 = layers.conv2d(l4, f5, 3, 2)
  return l5

def c2c(inflow, filters, name, kernel_size=3):
  f1, f2, f3 = filters
  with tf.variable_scope(name):
    l1 = layers.conv2d(inflow, f1, kernel_size)
    l2 = layers.conv2d(l1, f2, kernel_size)
    l3 = layers.conv2d(l2, f3, kernel_size, 2)
  return l3

def c5(inflow, outsize, name, filters=128):
  with tf.variable_scope(name):
    l1 = layers.conv2d(inflow, filters, 7)
    l2 = layers.conv2d(l1, filters, 7)
    l3 = layers.conv2d(l2, filters, 7)
    l4 = layers.conv2d(l3, filters, 1, activation_fn=None)
    l5 = layers.conv2d(l4, outsize, 1, activation_fn=None)
  return l5

def c7(inflow, outsize, name, filters=128):
  with tf.variable_scope(name):
    l1 = layers.conv2d(inflow, filters, 7)
    l2 = layers.conv2d(l1, filters, 7)
    l3 = layers.conv2d(l2, filters, 7)
    l4 = layers.conv2d(l3, filters, 7)
    l5 = layers.conv2d(l4, filters, 7)
    l6 = layers.conv2d(l5, filters, 1, activation_fn=None)
    l7 = layers.conv2d(l6, outsize, 1, activation_fn=None)
  return l7

def c7t(inflow, outsize, name, filters=128):
  with tf.variable_scope(name):
    l1 = layers.conv2d(inflow, filters, 7, activation_fn=tf.tanh)
    l2 = layers.conv2d(l1, filters, 7, activation_fn=tf.tanh)
    l3 = layers.conv2d(l2, filters, 7, activation_fn=tf.tanh)
    l4 = layers.conv2d(l3, filters, 7, activation_fn=tf.tanh)
    l5 = layers.conv2d(l4, filters, 7, activation_fn=tf.tanh)
    l6 = layers.conv2d(l5, filters, 1, activation_fn=tf.tanh)
    l7 = layers.conv2d(l6, outsize, 1, activation_fn=tf.tanh)
  return l7

def c7n(inflow, outsize, name, filters=128):
  with tf.variable_scope(name):
    l1 = layers.conv2d(inflow, filters, 7, activation_fn=None)
    l2 = layers.conv2d(l1, filters, 7, activation_fn=None)
    l3 = layers.conv2d(l2, filters, 7, activation_fn=None)
    l4 = layers.conv2d(l3, filters, 7, activation_fn=None)
    l5 = layers.conv2d(l4, filters, 7, activation_fn=None)
    l6 = layers.conv2d(l5, filters, 1, activation_fn=None)
    l7 = layers.conv2d(l6, outsize, 1, activation_fn=None)
  return l7

# use relu in all layers
def c7r(inflow, outsize, name, filters=128):
  with tf.variable_scope(name):
    l1 = layers.conv2d(inflow, filters, 7)
    l2 = layers.conv2d(l1, filters, 7)
    l3 = layers.conv2d(l2, filters, 7)
    l4 = layers.conv2d(l3, filters, 7)
    l5 = layers.conv2d(l4, filters, 7)
    l6 = layers.conv2d(l5, filters, 1)
    l7 = layers.conv2d(l6, outsize, 1)
  return l7

# use sigmoid in last layerer
def c7s(inflow, outsize, name, filters=128):
  with tf.variable_scope(name):
    l1 = layers.conv2d(inflow, filters, 7)
    l2 = layers.conv2d(l1, filters, 7)
    l3 = layers.conv2d(l2, filters, 7)
    l4 = layers.conv2d(l3, filters, 7)
    l5 = layers.conv2d(l4, filters, 7)
    l6 = layers.conv2d(l5, filters, 1)
    l7 = layers.conv2d(l6, outsize, 1, activation_fn=tf.sigmoid)
  return l7

def stage(inflow, name):
  kmap = c5(inflow, 14, name + '_1')
  amap = c5(inflow, 26, name + '_2')
  return kmap, amap

def stage7(inflow, name):
  kmap = c7(inflow, 14, name + '_1')
  amap = c7(inflow, 26, name + '_2')
  return kmap, amap

# affinity map
def a1():
  l0 = tf.placeholder(tf.float32, (None,None,None,3))

  # feature extraction
  l1 = c2(l0, 64, 'module_1')
  p1 = layers.max_pool2d(l1, 2)

  l2 = c2(p1, 128, 'module_2')
  p2 = layers.max_pool2d(l2, 2)

  l3 = c4(p2, (256,256,256,256), 'module_3')
  p3 = layers.max_pool2d(l3, 2)

  fmap = c4(p3, (512,512,256,256), 'module_4')

  l5_1 = c4(fmap, (128,128,128,512), 'stage_1_1')
  kmap_1 = layers.conv2d(l5_1, 14, 1) # 14 keypoints

  l5_2 = c4(fmap, (128,128,128,512), 'stage_1_2')
  amap_1 = layers.conv2d(l5_2, 26, 1, activation_fn=None) # 13 limbs

  concat_1 = tf.concat((kmap_1, amap_1, fmap), axis=3)

  kmap_2, amap_2 = stage(concat_1, 'stage_2')
  concat_2 = tf.concat((kmap_2, amap_2, fmap), axis=3)

  kmap_3, amap_3 = stage(concat_2, 'stage_3')
  concat_3 = tf.concat((kmap_3, amap_3, fmap), axis=3)

  kmap_4, amap_4 = stage(concat_3, 'stage_4')
  concat_4 = tf.concat((kmap_4, amap_4, fmap), axis=3)

  kmap_5, amap_5 = stage(concat_4, 'stage_5')
  concat_5 = tf.concat((kmap_5, amap_5, fmap), axis=3)

  kmap_6, amap_6 = stage(concat_5, 'stage_6')

  kmaps = [kmap_1, kmap_2, kmap_3, kmap_4, kmap_5, kmap_6]
  amaps = [amap_1, amap_2, amap_3, amap_4, amap_5, amap_6]

  return l0, kmaps, amaps

# use c7 replace c5
def a2():
  l0 = tf.placeholder(tf.float32, (None,None,None,3))

  # feature extraction
  l1 = c2(l0, 64, 'module_1')
  p1 = layers.max_pool2d(l1, 2)

  l2 = c2(p1, 128, 'module_2')
  p2 = layers.max_pool2d(l2, 2)

  l3 = c4(p2, (256,256,256,256), 'module_3')
  p3 = layers.max_pool2d(l3, 2)

  fmap = c4(p3, (512,512,256,256), 'module_4')

  l5_1 = c4(fmap, (128,128,128,512), 'stage_1_1')
  kmap_1 = layers.conv2d(l5_1, 14, 1) # 14 keypoints

  l5_2 = c4(fmap, (128,128,128,512), 'stage_1_2')
  amap_1 = layers.conv2d(l5_2, 26, 1, activation_fn=None) # 13 limbs

  concat_1 = tf.concat((kmap_1, amap_1, fmap), axis=3)

  kmap_2, amap_2 = stage7(concat_1, 'stage_2')
  concat_2 = tf.concat((kmap_2, amap_2, fmap), axis=3)

  kmap_3, amap_3 = stage7(concat_2, 'stage_3')
  concat_3 = tf.concat((kmap_3, amap_3, fmap), axis=3)

  kmap_4, amap_4 = stage7(concat_3, 'stage_4')
  concat_4 = tf.concat((kmap_4, amap_4, fmap), axis=3)

  kmap_5, amap_5 = stage7(concat_4, 'stage_5')
  concat_5 = tf.concat((kmap_5, amap_5, fmap), axis=3)

  kmap_6, amap_6 = stage7(concat_5, 'stage_6')

  kmaps = [kmap_1, kmap_2, kmap_3, kmap_4, kmap_5, kmap_6]
  amaps = [amap_1, amap_2, amap_3, amap_4, amap_5, amap_6]

  return l0, kmaps, amaps

# shallow
def dirmap():
  l0 = tf.placeholder(tf.float32, (None,368,368,3))

  # feature extraction
  l1 = c2(l0, 64, 'module_1')
  p1 = layers.max_pool2d(l1, 2)

  l2 = c2(p1, 128, 'module_2')
  p2 = layers.max_pool2d(l2, 2)

  l3 = c4(p2, (256,256,256,256), 'module_3')
  p3 = layers.max_pool2d(l3, 2)

  fmap = c4(p3, (512,512,256,256), 'module_4')

  l5_1 = c4(fmap, (128,128,128,512), 'stage_1_1')
  dmap_1 = layers.conv2d(l5_1, 52, 1, activation_fn=None) # 26*2 limbs

  concat_1 = tf.concat((dmap_1, fmap), axis=3)

  dmap_2 = c5(concat_1, 52, 'stage_2')
  concat_2 = tf.concat((dmap_2, fmap), axis=3)

  dmap_3 = c5(concat_2, 52, 'stage_3')
  concat_3 = tf.concat((dmap_3, fmap), axis=3)

  dmap_4 = c5(concat_3, 52, 'stage_4')
  # concat_4 = tf.concat((kmap_4, amap_4, fmap), axis=3)

  # kmap_5, amap_5 = stage(concat_4, 'stage_5')
  # concat_5 = tf.concat((kmap_5, amap_5, fmap), axis=3)

  # kmap_6, amap_6 = stage(concat_3, 'stage_6')

  dmaps = [dmap_1, dmap_2, dmap_3, dmap_4]

  return l0, dmaps

# deeper dmap
def v2():
  l0 = tf.placeholder(tf.float32, (None,368,368,3))

  # feature extraction
  l1 = c2(l0, 64, 'module_1')
  p1 = layers.max_pool2d(l1, 2)

  l2 = c2(p1, 128, 'module_2')
  p2 = layers.max_pool2d(l2, 2)

  l3 = c4(p2, (256,256,256,256), 'module_3')
  p3 = layers.max_pool2d(l3, 2)

  fmap = c4(p3, (512,512,256,256), 'module_4')

  l5_1 = c4(fmap, (128,128,128,512), 'stage_1_1')
  dmap_1 = layers.conv2d(l5_1, 52, 1) # 26*2 limbs

  concat_1 = tf.concat((dmap_1, fmap), axis=3)

  dmap_2 = c5(concat_1, 52, 'stage_2')
  concat_2 = tf.concat((dmap_2, fmap), axis=3)

  dmap_3 = c5(concat_2, 52, 'stage_3')
  concat_3 = tf.concat((dmap_3, fmap), axis=3)

  dmap_4 = c5(concat_3, 52, 'stage_4')
  concat_4 = tf.concat((dmap_4, fmap), axis=3)

  dmap_5 = c5(concat_4, 52, 'stage_5')
  concat_5 = tf.concat((dmap_5, fmap), axis=3)

  dmap_6 = c5(concat_5, 52, 'stage_6')

  dmaps = [dmap_1, dmap_2, dmap_3, dmap_4, dmap_5, dmap_6]

  return l0, dmaps

def fire(inflow, s11, e11, e33, name):
  with tf.variable_scope(name):
    squeeze = layers.conv2d(inflow, s11, 1)
    expand_1 = layers.conv2d(squeeze, e11, 1)
    expand_2 = layers.conv2d(squeeze, e33, 3)
    outflow = tf.concat((expand_1, expand_2), axis=3)
  return outflow

# use fire
def v3():
  l0 = tf.placeholder(tf.float32, (None,368,368,3))

  # feature extraction
  l1 = layers.conv2d(l0, 64, 7)
  p1 = layers.max_pool2d(l1, 2)

  l2 = fire(p1, 32, 128, 128, 'module_2')
  p2 = layers.max_pool2d(l2, 2)

  l3 = fire(p2, 48, 192, 192, 'module_3')
  p3 = layers.max_pool2d(l3, 2)

  l4 = fire(p3, 64, 256, 256, 'module_4')
  fmap = layers.conv2d(l4, 256, 1)

  l5_1 = c4(fmap, (128,128,128,512), 'stage_1')
  dmap_1 = layers.conv2d(l5_1, 52, 1, activation_fn=None) # 26*2 limbs

  concat_1 = tf.concat((dmap_1, fmap), axis=3)

  dmap_2 = c5(concat_1, 52, 'stage_2')
  concat_2 = tf.concat((dmap_2, fmap), axis=3)

  dmap_3 = c5(concat_2, 52, 'stage_3')
  concat_3 = tf.concat((dmap_3, fmap), axis=3)

  dmap_4 = c5(concat_3, 52, 'stage_4')
  concat_4 = tf.concat((dmap_4, fmap), axis=3)

  dmap_5 = c5(concat_4, 52, 'stage_5')
  concat_5 = tf.concat((dmap_5, fmap), axis=3)

  dmap_6 = c5(concat_5, 52, 'stage_6')

  dmaps = [dmap_1, dmap_2, dmap_3, dmap_4, dmap_5, dmap_6]

  return l0, dmaps

# fix dmap_1 bug(relu)
def v4():
  l0 = tf.placeholder(tf.float32, (None,368,368,3))

  # feature extraction
  l1 = c2(l0, 64, 'module_1')
  p1 = layers.max_pool2d(l1, 2)

  l2 = c2(p1, 128, 'module_2')
  p2 = layers.max_pool2d(l2, 2)

  l3 = c4(p2, (256,256,256,256), 'module_3')
  p3 = layers.max_pool2d(l3, 2)

  fmap = c4(p3, (512,512,256,256), 'module_4')

  l5_1 = c4(fmap, (128,128,128,512), 'stage_1_1')
  dmap_1 = layers.conv2d(l5_1, 52, 1, activation_fn=None) # 26*2 limbs

  concat_1 = tf.concat((dmap_1, fmap), axis=3)

  dmap_2 = c5(concat_1, 52, 'stage_2')
  concat_2 = tf.concat((dmap_2, fmap), axis=3)

  dmap_3 = c5(concat_2, 52, 'stage_3')
  concat_3 = tf.concat((dmap_3, fmap), axis=3)

  dmap_4 = c5(concat_3, 52, 'stage_4')
  concat_4 = tf.concat((dmap_4, fmap), axis=3)

  dmap_5 = c5(concat_4, 52, 'stage_5')
  concat_5 = tf.concat((dmap_5, fmap), axis=3)

  dmap_6 = c5(concat_5, 52, 'stage_6')

  dmaps = [dmap_1, dmap_2, dmap_3, dmap_4, dmap_5, dmap_6]

  return l0, dmaps

# deeper fmap
def v7():
  l0 = tf.placeholder(tf.float32, (None,368,368,3))

  # feature extraction
  l1 = c4(l0, (64,64,64,64), 'module_1')
  p1 = layers.max_pool2d(l1, 2)

  l2 = c4(p1, (128,128,128,128), 'module_2')
  p2 = layers.max_pool2d(l2, 2)

  l3 = c4(p2, (256,256,256,256), 'module_3')
  p3 = layers.max_pool2d(l3, 2)

  fmap = c4(p3, (512,512,256,256), 'module_4')

  l5_1 = c4(fmap, (128,128,128,512), 'stage_1_1')
  dmap_1 = layers.conv2d(l5_1, 52, 1, activation_fn=None) # 26*2 limbs

  concat_1 = tf.concat((dmap_1, fmap), axis=3)

  dmap_2 = c5(concat_1, 52, 'stage_2')
  concat_2 = tf.concat((dmap_2, fmap), axis=3)

  dmap_3 = c5(concat_2, 52, 'stage_3')
  concat_3 = tf.concat((dmap_3, fmap), axis=3)

  dmap_4 = c5(concat_3, 52, 'stage_4')
  concat_4 = tf.concat((dmap_4, fmap), axis=3)

  dmap_5 = c5(concat_4, 52, 'stage_5')
  concat_5 = tf.concat((dmap_5, fmap), axis=3)

  dmap_6 = c5(concat_5, 52, 'stage_6')

  dmaps = [dmap_1, dmap_2, dmap_3, dmap_4, dmap_5, dmap_6]

  return l0, dmaps

# single direction: forward or backward
def v8():
  l0 = tf.placeholder(tf.float32, (None,368,368,3))

  # feature extraction
  l1 = c4(l0, (64,64,64,64), 'module_1')
  p1 = layers.max_pool2d(l1, 2)

  l2 = c4(p1, (128,128,128,128), 'module_2')
  p2 = layers.max_pool2d(l2, 2)

  l3 = c4(p2, (256,256,256,256), 'module_3')
  p3 = layers.max_pool2d(l3, 2)

  fmap = c4(p3, (512,512,256,256), 'module_4')

  l5_1 = c4(fmap, (128,128,128,512), 'stage_1_1')
  dmap_1 = layers.conv2d(l5_1, 26, 1, activation_fn=None) # 26*2 limbs

  concat_1 = tf.concat((dmap_1, fmap), axis=3)

  dmap_2 = c5(concat_1, 26, 'stage_2')
  concat_2 = tf.concat((dmap_2, fmap), axis=3)

  dmap_3 = c5(concat_2, 26, 'stage_3')
  concat_3 = tf.concat((dmap_3, fmap), axis=3)

  dmap_4 = c5(concat_3, 26, 'stage_4')
  concat_4 = tf.concat((dmap_4, fmap), axis=3)

  dmap_5 = c5(concat_4, 26, 'stage_5')
  concat_5 = tf.concat((dmap_5, fmap), axis=3)

  dmap_6 = c5(concat_5, 26, 'stage_6')

  dmaps = [dmap_1, dmap_2, dmap_3, dmap_4, dmap_5, dmap_6]

  return l0, dmaps

def v9():
  l0 = tf.placeholder(tf.float32, (None,368,368,3))

  # feature extraction
  l1 = c4c(l0, (64,64,64,64,64), 'module_1')

  l2 = c4c(l1, (128,128,128,128,128), 'module_2')

  l3 = c4c(l2, (256,256,256,256,256), 'module_3')

  fmap = c4(l3, (512,512,256,256), 'module_4')

  l5_1 = c4(fmap, (128,128,128,512), 'stage_1_1')
  dmap_1 = layers.conv2d(l5_1, 52, 1, activation_fn=None) # 26*2 limbs

  concat_1 = tf.concat((dmap_1, fmap), axis=3)

  dmap_2 = c5(concat_1, 52, 'stage_2')
  concat_2 = tf.concat((dmap_2, fmap), axis=3)

  dmap_3 = c5(concat_2, 52, 'stage_3')
  concat_3 = tf.concat((dmap_3, fmap), axis=3)

  dmap_4 = c5(concat_3, 52, 'stage_4')
  concat_4 = tf.concat((dmap_4, fmap), axis=3)

  dmap_5 = c5(concat_4, 52, 'stage_5')
  concat_5 = tf.concat((dmap_5, fmap), axis=3)

  dmap_6 = c5(concat_5, 52, 'stage_6')
  concat_6 = tf.concat((dmap_6, fmap), axis=3)

  dmap_tiny = layers.conv2d_transpose(concat_6, 52, 9, 2)
  fmap_tiny = layers.conv2d_transpose(fmap, 256, 3, 2)
  concat_tiny = tf.concat((dmap_tiny, fmap_tiny), axis=3)

  dmap_mid = layers.conv2d_transpose(concat_tiny, 52, 9, 2)
  fmap_mid = layers.conv2d_transpose(fmap_tiny, 256, 3, 2)
  concat_mid = tf.concat((dmap_mid, fmap_mid), axis=3)

  dmap_large = layers.conv2d_transpose(concat_mid, 52, 9, 2)

  dmaps = [dmap_1, dmap_2, dmap_3, dmap_4, dmap_5, dmap_6, dmap_tiny, dmap_mid, dmap_large]

  return l0, dmaps

# use deeper dmap(c7 replace c5)
def v10():
  l0 = tf.placeholder(tf.float32, (None,None,None,3))

  # feature extraction
  l1 = c2c(l0, (64,64,64), 'module_1')

  l2 = c2c(l1, (128,128,128), 'module_2')

  l3 = c4c(l2, (256,256,256,256,256), 'module_3')

  fmap = c4(l3, (512,512,256,256), 'module_4')

  l5_1 = c4(fmap, (128,128,128,512), 'stage_1_1')
  dmap_1 = layers.conv2d(l5_1, 52, 1, activation_fn=None) # 26*2 limbs

  concat_1 = tf.concat((dmap_1, fmap), axis=3)

  dmap_2 = c7(concat_1, 52, 'stage_2')
  concat_2 = tf.concat((dmap_2, fmap), axis=3)

  dmap_3 = c7(concat_2, 52, 'stage_3')
  concat_3 = tf.concat((dmap_3, fmap), axis=3)

  dmap_4 = c7(concat_3, 52, 'stage_4')
  concat_4 = tf.concat((dmap_4, fmap), axis=3)

  dmap_5 = c7(concat_4, 52, 'stage_5')
  concat_5 = tf.concat((dmap_5, fmap), axis=3)

  dmap_6 = c7(concat_5, 52, 'stage_6')

  dmaps = [dmap_1, dmap_2, dmap_3, dmap_4, dmap_5, dmap_6]

  return l0, dmaps

# all use tanh in dmap
def v13():
  l0 = tf.placeholder(tf.float32, (None,None,None,3))

  # feature extraction
  l1 = c2c(l0, (64,64,64), 'module_1')

  l2 = c2c(l1, (128,128,128), 'module_2')

  l3 = c4c(l2, (256,256,256,256,256), 'module_3')

  fmap = c4(l3, (512,512,256,256), 'module_4')

  l5_1 = c4(fmap, (128,128,128,512), 'stage_1_1')
  dmap_1 = layers.conv2d(l5_1, 52, 1, activation_fn=tf.tanh) # 26*2 limbs

  concat_1 = tf.concat((dmap_1, fmap), axis=3)

  dmap_2 = c7t(concat_1, 52, 'stage_2')
  concat_2 = tf.concat((dmap_2, fmap), axis=3)

  dmap_3 = c7t(concat_2, 52, 'stage_3')
  concat_3 = tf.concat((dmap_3, fmap), axis=3)

  dmap_4 = c7t(concat_3, 52, 'stage_4')
  concat_4 = tf.concat((dmap_4, fmap), axis=3)

  dmap_5 = c7t(concat_4, 52, 'stage_5')
  concat_5 = tf.concat((dmap_5, fmap), axis=3)

  dmap_6 = c7t(concat_5, 52, 'stage_6')

  dmaps = [dmap_1, dmap_2, dmap_3, dmap_4, dmap_5, dmap_6]

  return l0, dmaps

# all use None in dmap
def v14():
  l0 = tf.placeholder(tf.float32, (None,None,None,3))

  # feature extraction
  l1 = c2c(l0, (64,64,64), 'module_1')

  l2 = c2c(l1, (128,128,128), 'module_2')

  l3 = c4c(l2, (256,256,256,256,256), 'module_3')

  fmap = c4(l3, (512,512,256,256), 'module_4')

  l5_1 = c4(fmap, (128,128,128,512), 'stage_1_1')
  dmap_1 = layers.conv2d(l5_1, 52, 1, activation_fn=tf.tanh) # 26*2 limbs

  concat_1 = tf.concat((dmap_1, fmap), axis=3)

  dmap_2 = c7n(concat_1, 52, 'stage_2')
  concat_2 = tf.concat((dmap_2, fmap), axis=3)

  dmap_3 = c7n(concat_2, 52, 'stage_3')
  concat_3 = tf.concat((dmap_3, fmap), axis=3)

  dmap_4 = c7n(concat_3, 52, 'stage_4')
  concat_4 = tf.concat((dmap_4, fmap), axis=3)

  dmap_5 = c7n(concat_4, 52, 'stage_5')
  concat_5 = tf.concat((dmap_5, fmap), axis=3)

  dmap_6 = c7n(concat_5, 52, 'stage_6')

  dmaps = [dmap_1, dmap_2, dmap_3, dmap_4, dmap_5, dmap_6]

  return l0, dmaps

def res_block(inflow, scope):
  with tf.variable_scope(scope):
    l1 = layers.conv2d(inflow, 64, 1)
    l2 = layers.conv2d(l1, 64, 3)
    l3 = layers.conv2d(l2, 256, 1)
    l4 = l3 + inflow
    l5 = tf.nn.relu(l4)
  return l5

# predict only kmap, use res_block
def k1():
  l0 = tf.placeholder(tf.float32, (None,None,None,3))
  training = tf.placeholder(tf.bool)

  l1 = layers.conv2d(l0, 256, 7, 2)
  l2 = res_block(l1, 'block_1')
  l3 = layers.conv2d(l2, 256, 7, 2)
  l4 = res_block(l3, 'block_2')
  l5 = layers.conv2d(l4, 256, 7, 2)
  fmap = res_block(l5, 'block_3')
  k1 = layers.conv2d(fmap, 14, 7)
  c1 = tf.concat((k1, fmap), axis=3)
  k2 = c7r(c1, 14, 'stage_1')
  c2 = tf.concat((k2, fmap), axis=3)
  k3 = c7r(c2, 14, 'stage_2')
  c3 = tf.concat((k3, fmap), axis=3)
  k4 = c7r(c3, 14, 'stage_3')
  c4 = tf.concat((k4, fmap), axis=3)
  k5 = c7r(c4, 14, 'stage_4')
  c5 = tf.concat((k5, fmap), axis=3)
  k6 = c7r(c5, 14, 'stage_5')

  kmaps = [k1, k2, k3, k4, k5, k6]

  return l0, kmaps, fmap

# use traditional conv
def k2():
  l0 = tf.placeholder(tf.float32, (None,None,None,3))

  # feature extraction
  l1 = c2c(l0, (64,64,64), 'module_1')
  l2 = c2c(l1, (128,128,128), 'module_2')
  l3 = c4c(l2, (256,256,256,256,256), 'module_3')
  fmap = c4(l3, (512,512,256,256), 'module_4')
  l5_1 = c4(fmap, (128,128,128,512), 'stage_1_1')
  kmap_1 = layers.conv2d(l5_1, 14, 1, activation_fn=tf.sigmoid) # 26*2 limbs
  concat_1 = tf.concat((kmap_1, fmap), axis=3)
  kmap_2 = c7s(concat_1, 14, 'stage_2')
  concat_2 = tf.concat((kmap_2, fmap), axis=3)
  kmap_3 = c7s(concat_2, 14, 'stage_3')
  concat_3 = tf.concat((kmap_3, fmap), axis=3)
  kmap_4 = c7s(concat_3, 14, 'stage_4')
  concat_4 = tf.concat((kmap_4, fmap), axis=3)
  kmap_5 = c7s(concat_4, 14, 'stage_5')
  concat_5 = tf.concat((kmap_5, fmap), axis=3)
  kmap_6 = c7s(concat_5, 14, 'stage_6')

  kmaps = [kmap_1, kmap_2, kmap_3, kmap_4, kmap_5, kmap_6]

  return l0, kmaps

def l1():
  img = tf.placeholder(tf.float32, (None, 368, 368, 3))
  dmap = tf.placeholder(tf.float32, (None, 46, 46, 52))
  kmap = tf.placeholder(tf.float32, (None, 46, 46, 2))

  l1 = c2c(img, (64,64,64), 'module_1', 5)
  l2 = c2c(l1, (128,128,128), 'module_2', 5)
  l3 = c2c(l2, (128,128,128), 'module_3', 5)

  c1 = tf.concat((l3, dmap, kmap), axis=3)

  l4 = c2c(c1, (256,256,256), 'module_4', 5)
  l5 = c2c(l4, (256,256,256), 'module_5', 5)
  l6 = c2c(l5, (256,256,256), 'module_6', 5)
  l7 = layers.flatten(l6)
  l8 = layers.fully_connected(l7, 1, activation_fn=tf.sigmoid)

  return img, dmap, kmap, l8

def compute_k_loss(inflow, l_rate):
  ref = tf.placeholder(tf.float32, inflow[0].shape)
  loss = tf.constant(0, dtype=tf.float32)
  for m in inflow:
    loss += tf.reduce_mean(tf.reduce_sum(tf.square(m - ref), axis=(1,2,3)))
  avg_loss = loss / len(inflow)
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_step = tf.train.AdagradOptimizer(l_rate).minimize(avg_loss)
  return ref, avg_loss, train_step

def compute_single_loss(inflow):
  ref = tf.placeholder(tf.float32, inflow[0].shape)
  loss = tf.constant(0, dtype=tf.float32)
  for m in inflow:
    loss += tf.reduce_mean(tf.reduce_sum(tf.square(m - ref), axis=(1,2,3)))

  return ref, loss

def compute_loss(kmaps, amaps, ratio=0.5):
  ref_kmap = tf.placeholder(tf.float32, kmaps[0].shape)
  ref_amap = tf.placeholder(tf.float32, amaps[0].shape)

  k_loss = tf.constant(0, dtype=tf.float32)
  a_loss = tf.constant(0, dtype=tf.float32)
  for m in kmaps:
    k_loss += tf.reduce_mean(tf.reduce_sum(tf.square(m - ref_kmap), axis=(1,2,3)))
  for m in amaps:
    a_loss += tf.reduce_mean(tf.reduce_sum(tf.square(m - ref_amap), axis=(1,2,3)))
  loss = k_loss + ratio * a_loss

  return ref_kmap, ref_amap, k_loss, a_loss, loss

if __name__ == '__main__':
  l0, kmaps, amaps = vanilla()
  print('kmaps:')
  for m in kmaps:
    print(m.get_shape())
  print('amaps:')
  for m in amaps:
    print(m.get_shape())

