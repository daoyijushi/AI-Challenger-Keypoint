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

def c7(inflow, outsize, name, filters=128):
  with tf.variable_scope(name):
    l1 = layers.conv2d(inflow, filters, 7)
    l2 = layers.conv2d(l1, filters, 7)
    l3 = layers.conv2d(l2, filters, 7)
    # l4 = layers.conv2d(l3, filters, 7)
    # l5 = layers.conv2d(l4, filters, 7)
    l6 = layers.conv2d(l3, filters, 1)
    l7 = layers.conv2d(l6, outsize, 1)
  return l7

def stage(inflow, name):
  kmap = c7(inflow, 14, name + '_1')
  amap = c7(inflow, 26, name + '_2')
  return kmap, amap

def vanilla():
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
  kmap_1 = layers.conv2d(l5_1, 14, 1) # 14 keypoints

  l5_2 = c4(fmap, (128,128,128,512), 'stage_1_2')
  amap_1 = layers.conv2d(l5_2, 26, 1) # 13 limbs

  concat_1 = tf.concat((kmap_1, amap_1, fmap), axis=3)

  kmap_2, amap_2 = stage(concat_1, 'stage_2')
  concat_2 = tf.concat((kmap_2, amap_2, fmap), axis=3)

  kmap_3, amap_3 = stage(concat_2, 'stage_3')
  concat_3 = tf.concat((kmap_3, amap_3, fmap), axis=3)

  # kmap_4, amap_4 = stage(concat_3, 'stage_4')
  # concat_4 = tf.concat((kmap_4, amap_4, fmap), axis=3)

  # kmap_5, amap_5 = stage(concat_4, 'stage_5')
  # concat_5 = tf.concat((kmap_5, amap_5, fmap), axis=3)

  kmap_6, amap_6 = stage(concat_3, 'stage_6')

  kmaps = [kmap_1, kmap_2, kmap_3, kmap_6]
  amaps = [amap_1, amap_2, amap_3, amap_6]

  return l0, kmaps, amaps

def compute_loss(kmaps, amaps, ratio=0.01):
  ref_kmap = tf.placeholder(tf.float32, (None,46,46,14))
  ref_amap = tf.placeholder(tf.float32, (None,46,46,26))

  k_loss = tf.zeros([1])
  a_loss = tf.zeros([1])
  for m in kmaps:
    k_loss += tf.reduce_mean(tf.square(m - ref_kmap))
  for m in amaps:
    a_loss += tf.reduce_mean(tf.square(m - ref_amap))
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

