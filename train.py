import network
import reader
import util
import tensorflow as tf
import time
import numpy as np
from scipy import misc

r = reader.Reader('./data/train/', 'annotations.pkl', 32)
l_rate = 1e-1

inflow, kmaps, amaps = network.vanilla()
ref_kmap, ref_amap, k_loss, a_loss, loss = network.compute_loss(kmaps, amaps)
train_step = tf.train.AdagradOptimizer(l_rate).minimize(loss)

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

for i in range(1000000):
  tic = time.time()
  img, kmap, amap = r.next_batch()
  img = np.zeros((32,368,368,3))
  _, batch_k_loss, batch_a_loss = \
    sess.run([train_step, k_loss, a_loss], feed_dict={inflow:img, ref_kmap:kmap, ref_amap:amap})
  toc = time.time()
  interval = (toc - tic) * 1000
  r.index = 0 # train on the first batch for sanity check

  print('Iter %d, k loss %g, a loss %g, timecost %g ms' % \
    (i, batch_k_loss, batch_a_loss, interval))
  
  if i % 1000 == 0:
    k, a = sess.run([kmaps,amaps], feed_dict={inflow:img[0:1]})
    img = np.zeros([46,46,3])
    k = k[0].reshape([46,46,14])
    a = a[0].reshape([46,46,26])
    util.visualization(img, k, a, str(i)+'.jpg')
