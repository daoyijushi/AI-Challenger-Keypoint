import network_simp
import reader
import util
import tensorflow as tf
import time
import numpy as np
from scipy import misc

r = reader.Reader('./data/train/', 'annotations.pkl', 32)
l_rate = 1e-3

inflow, kmaps= network_simp.vanilla()
ref_kmap, loss = network_simp.compute_loss(kmaps)
train_step = tf.train.AdagradOptimizer(l_rate).minimize(loss)

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

for i in range(1000000):
  tic = time.time()
  img, kmap, amap = r.next_batch()
  r.index = 0 # train on the first batch for sanity check
  _, batch_loss = \
    sess.run([train_step, loss], feed_dict={inflow:img, ref_kmap:kmap})
  toc = time.time()
  interval = (toc - tic) * 1000

  print('Iter %d, loss %g, timecost %g ms' % (i, batch_loss, interval))
  
  if i % 1000 == 0:
    k = sess.run(kmaps, feed_dict={inflow:img[0:1]})[0].reshape([46,46,14])
    k = np.amax(k, axis=2)
    misc.imsave("%d.jpg" % i, k)
