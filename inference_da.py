import network
import tensorflow as tf
import time
import gflags
import sys
import os
import scipy.misc as misc
import json
import util
import numpy as np

Flags = gflags.FLAGS

gflags.DEFINE_string('model_path', '', 'where to restore model')
gflags.DEFINE_string('test_path', '', 'where are the test images')
gflags.DEFINE_string('save_path', '', 'where to save the results')

Flags(sys.argv)

model_path = Flags.model_path
test_path = Flags.test_path
save_path = Flags.save_path

names = os.listdir(test_path)

inflow, kmaps, amaps = network.a2()

sess = tf.Session()
saver = tf.train.Saver()

ckpt = tf.train.get_checkpoint_state(model_path)
if ckpt:
  saver.restore(sess, ckpt.model_checkpoint_path)
  print(ckpt.model_checkpoint_path)
else:
  print('No available ckpt.')

limbs = util.get_limbs()
connections = util.get_connections()
patch = util.get_patch(10, 4)
result = []
mean = np.array([122.35131039, 115.17054545, 107.60200075])
var = np.array([35.77071304, 35.39201422, 37.7260754])
for name in names:
  tic = time.time()
  src = misc.imread(test_path+name)
  imgs, lefts, tops, rate = util.multi_resize(src, 368, 24)

  imgs -= mean
  imgs /= var

  rate /= 8 # due to dowsn sampling
  batch_k, batch_a = sess.run([kmaps, amaps], feed_dict={inflow:imgs})

  batch_k = batch_k[-1]
  batch_a = batch_a[-1]

  k = util.concat_maps(batch_k, lefts, tops, 8)
  a = util.concat_maps(batch_a, lefts, tops, 8)

  np.save('amap.npy', a)
  np.save('kmap.npy', k)

  util.vis_kmap(k, 'kmap.jpg')
  util.vis_amap(a, 'amap.jpg')


  # humans = inf_util.reconstruct(dmap, kmap, 5)
  # annos = inf_util.format(humans, name.split('.')[0], rate)
  # result.append(annos)

#   h, w, _ = dmap.shape
#   grid_h, grid_w = util.get_grid(h, w)

#   kmap = util.get_kmap_from_dmap(dmap, limbs)
#   # util.vis_kmap(kmap, name.split('.')[0]+('_big.jpg'))

#   annos = util.rebuild(dmap, kmap, connections, 2, grid_h, grid_w, patch, rate)
#   result.append(util.format_annos(annos, name.split('.')[0]))
  toc = time.time()

  print(name, 'time cost', toc-tic)

# j = json.dumps(result)
# with open(save_path, 'w') as f:
#   f.write(j)


