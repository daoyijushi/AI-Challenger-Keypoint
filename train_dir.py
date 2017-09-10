import network
import reader
import util
import tensorflow as tf
import time
import numpy as np
from scipy import misc

r = reader.DirReader('./data/train/', 'annotations.pkl', 32)
l_rate = 1e-3

inflow, dmaps = network.dirmap()
ref_dmap, loss = network.compute_single_loss(dmaps)
train_step = tf.train.AdagradOptimizer(l_rate).minimize(loss)

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

for i in range(1000000):
  tic = time.time()
  img, kmap, dmap = r.next_batch()
  r.index = 0 # train on the first batch for sanity check
  _, batch_loss = \
    sess.run([train_step, loss], feed_dict={inflow:img, ref_dmap:dmap})
  toc = time.time()
  interval = (toc - tic) * 1000

  print('Iter %d, loss %g, timecost %g ms' % (i, batch_loss, interval))
  
  if i % 1000 == 0:
    d = sess.run(dmaps, feed_dict={inflow:img[0:1]})[0]
    d = d[0].reshape([46,46,52])
    dx = d[:,:,::2]
    dy = d[:,:,1::2]
    d = np.square(dx) + np.square(dy)
    d = np.max(d, axis=2)
    misc.imsave('dir_%d.jpg' % i, d)
