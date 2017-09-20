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
r = reader.DirReader('./data/train/', 'annotations_new.pkl', 16)

sess = tf.Session()

inflow, dmaps = network.v12()
ref_dmap, loss = network.compute_single_loss(dmaps)
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
  img, _, dmap, names = r.next_batch()

  _, batch_loss, step_cnt, log = \
    sess.run([train_step, loss, one_step_op, merged], feed_dict={inflow:img, ref_dmap:dmap})

  toc = time.time()
  interval = (toc - tic) * 1000

  print('Iter %d, loss %g, timecost %g ms' % (step_cnt, batch_loss, interval))
  writer.add_summary(log, step_cnt)

  if step_cnt % 500 == 0 or step_cnt == 1:
    save_name = '%s.ckpt' % model_name
    saver.save(sess, model_path+save_name, global_step=step_cnt)

    d = sess.run(dmaps, feed_dict={inflow:img[0:1]})
    d = d[-1].reshape([46,46,52])
    util.vis_dmap(d, 'pred_%d.jpg' % step_cnt)
    util.vis_dmap(dmap[0], 'truth_%d.jpg' % step_cnt)
    misc.imsave('src_%d.jpg' % step_cnt, img[0])
    with open('train_log.txt', 'a') as f:
      f.write(str(step_cnt))
      f.write(' ')
      f.write(names[0])
      f.write('\n')
    



