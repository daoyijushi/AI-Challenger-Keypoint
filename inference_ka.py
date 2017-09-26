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
import inf_util_ka

Flags = gflags.FLAGS

gflags.DEFINE_string('model_path', '', 'where to restore model')
gflags.DEFINE_string('test_path', '', 'where are the test images')
gflags.DEFINE_string('save_path', '', 'where to save the results')
gflags.DEFINE_bool('vis_anno', False, 'visualize the annotations or not')
gflags.DEFINE_bool('save', False, 'save kmap and amap')
gflags.DEFINE_bool('use_old', False, 'use old version')

Flags(sys.argv)

model_path = Flags.model_path
test_path = Flags.test_path
save_path = Flags.save_path
vis_anno = Flags.vis_anno
use_old = Flags.use_old
save = Flags.save

names = os.listdir(test_path)

inflow, kmaps, amaps = network.a4()

sess = tf.Session()
saver = tf.train.Saver()

if use_old:
  try:
    saver.restore(sess, model_path)
    print(model_path)
  except Exception as e:
    print(e)
    exit(0)
else:
  ckpt = tf.train.get_checkpoint_state(model_path)
  if ckpt:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print(ckpt.model_checkpoint_path)
  else:
    print('No available ckpt.')
    exit(0)

limbs = util.get_limbs()
connections = util.get_connections()
patch = util.get_patch(10, 4)
result = []
mean = np.array([122.35131039, 115.17054545, 107.60200075])
var = np.array([35.77071304, 35.39201422, 37.7260754])

cnt = 0
total = len(names)
elapse = 0

for name in names:
  tic = time.time()
  src = misc.imread(test_path+name)
  h, w, _ = src.shape
  imgs, lefts, tops, rate = util.multi_resize(src, 368, 24)

  imgs -= mean
  imgs /= var

  rate /= 8 # due to dowsn sampling
  batch_k, batch_a = sess.run([kmaps, amaps], feed_dict={inflow:imgs})

  batch_k = batch_k[-1]
  batch_a = batch_a[-1]

  k = util.concat_maps(batch_k, lefts, tops, 8)
  a = util.concat_maps(batch_a, lefts, tops, 8)

  humans = inf_util_ka.reconstruct(a, k, 10)
  annos = inf_util_ka.format(humans, name.split('.')[0], rate)
  result.append(annos)

  if vis_anno:
    for human, ann in annos['keypoint_annotations'].items():
      k_rev = util.get_key_hmap((h, w), [ann], util.get_patch(40,64), r=20)
      src = misc.imread(test_path+name)
      util.cover_key_map(src, k_rev)
      misc.imsave('%s_%s.jpg'%(annos['image_id'], human), src)

  if save:
    np.save(annos['image_id']+'_k.npy', k)
    np.save(annos['image_id']+'_a.npy', a)

  toc = time.time()

  cnt += 1
  interval = toc - tic
  elapse += interval
  if cnt == 2:
    elapse = interval * 2
  remain = (elapse/cnt)*(total-cnt)
  print('%d/%d time cost %g remain %g' % (cnt, total, interval, remain))


j = json.dumps(result)
with open(save_path, 'w') as f:
  f.write(j)


