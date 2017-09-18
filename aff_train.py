import network
import reader
import util
import tensorflow as tf
import time
import numpy as np
import scipy.misc as misc
import sys
import os

model_name = sys.argv[1]
model_path = './model/' + model_name + '/'

l_rate = float(sys.argv[2])
r = reader.Reader('./data/train/', 'annotations_new.pkl', 16)

sess = tf.Session()

inflow, kmaps, amaps = network.vanilla()
k_ref, a_ref, k_loss, a_loss, loss = network.compute_loss(kmaps, amaps, 0.5)
train_step = tf.train.AdagradOptimizer(l_rate).minimize(loss)

tf.summary.scalar('loss', loss)
tf.summary.scalar('learning rate', l_rate)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(model_path, sess.graph)

global_step = tf.Variable(0, name='global_step', trainable=False)
one_step_op = global_step.assign_add(1)

init_op = tf.global_variables_initializer()
sess.run(init_op)

saver = tf.train.Saver()
if os.path.exists(model_path):
  ckpt = tf.train.get_checkpoint_state(model_path)
  if ckpt:
    saver.restore(sess, ckpt.model_checkpoint_path)

while True:
  tic = time.time()
  img, keypoint_hmap, affinity_hmap, names = r.next_batch()
  fd = {
    inflow: img, 
    k_ref: keypoint_hmap,
    a_ref: affinity_hmap
  }

  _, batch_k_loss, batch_a_loss, batch_loss, step_cnt, log = \
    sess.run([train_step, k_loss, a_loss, loss, one_step_op, merged], feed_dict=fd)

  toc = time.time()
  interval = (toc - tic) * 1000

  print('Iter %d, k loss %g, a loss %g, total loss %g, timecost %g ms' % (step_cnt, batch_k_loss, batch_a_loss, batch_loss, interval))
  writer.add_summary(log, step_cnt)

  if step_cnt % 500 == 0 or step_cnt == 1:
    save_name = '%s.ckpt' % model_name
    saver.save(sess, model_path+save_name, global_step=step_cnt)

    k, a = sess.run([kmaps, amaps], feed_dict={inflow:img[0:1]})
    k = k[-1].reshape((46,46,14))
    a = a[-1].reshape((46,46,26))
    tmp = misc.imresize(img[0], (46,46))
    util.visualization(tmp, k, a, 'pred_%d.jpg'%step_cnt)
    util.visualization(tmp, keypoint_hmap[0], affinity_hmap[0], 'truth_%d.jpg'%step_cnt)
    misc.imsave('src_%d.jpg'%step_cnt, img[0])
    with open('train_log.txt', 'a') as f:
      f.write(str(int(step_cnt)))
      f.write(' ')
      f.write(names[0])
      f.write('\n')
    



