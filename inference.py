import network
import tensorflow as tf
import time
import gflags
import sys
import os
import scipy.misc as misc
import json

Flags = gflags.FLAGS

gflags.DEFINE_string('model_path', '', 'where to restore model')
gflags.DEFINE_string('test_path', '', 'where are the test images')
gflags.DEFINE_string('save_path', '', 'where to save the results')

Flags(sys.argv)

model_path = Flags.model_path
test_path = Flags.test_path
save_path = Flags.save_path

names = os.listdir(test_path)

inflow, dmaps = network.v4()

sess = tf.Session
saver = tf.train.Saver()

ckpt = tf.train.get_checkpoint_state(model_path)
if ckpt:
  saver.restore(sess, ckpt.model_checkpoint_path)
else:
  print('No available ckpt.')

limbs = util.get_limbs()
connections = util.get_connections()
grid = util.get_grid(46)
patch = util.get_patch(10, 4)
result = []

for name in names:
  src = misc.imread(test_path+name)
  imgs, lefts, tops, rate = util.multi_resize(src, 368, 24)
  rate /= 8 # due to pooling
  batch_dmaps = sess.run(dmaps, feed_dict={inflow:imgs})[-1]
  dmap = util.concat_dmaps(batch_dmaps, img2dmap)
  kmap = util.get_kmap_from_dmap(dmap, limbs)
  annos = util.rebuild(dmap, kmap, connections, 2, grid, patch, rate)
  result.append(util.format_annos(annos, name.split('.')[0]))

j = json.dumps(result)
with open('inference_result.json', 'w') as f:
  f.write(j)

