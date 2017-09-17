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

step_cnt = int(sys.argv[2])
l_rate = float(sys.argv[3])
reader_type = sys.argv[4]

r = None
if reader_type == 'dir':
  r = reader.DirReader('./data/train/', 'annotations_new.pkl', 16)
elif reader_type == 'forward':
  r = reader.ForwardReader('./data/train/', 'annotations_new.pkl', 16)
elif reader_type == 'backward':
  r = reader.BackwardReader('./data/train/', 'annotations_new.pkl', 16)
else:
  print('Error reader.')
  exit(0)

sess = tf.Session()

inflow, dmaps = network.v7()
ref_dmap, loss = network.compute_single_loss(dmaps)
train_step = tf.train.AdagradOptimizer(l_rate).minimize(loss)

tf.summary.scalar('loss', loss)
tf.summary.scalar('learning rate', l_rate)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(model_path, sess.graph)

saver = tf.train.Saver()

init_op = tf.global_variables_initializer()
sess.run(init_op)

if os.path.exists(model_path):
  ckpt = tf.train.get_checkpoint_state(model_path)
  if ckpt:
    saver.restore(sess, ckpt.model_checkpoint_path)


i = step_cnt
while True:

  tic = time.time()
  img, kmap, dmap, names = r.next_batch()
  _, batch_loss, log = \
    sess.run([train_step, loss, merged], feed_dict={inflow:img, ref_dmap:dmap})
  toc = time.time()
  interval = (toc - tic) * 1000

  print('Iter %d, loss %g, timecost %g ms' % (i, batch_loss, interval))
  writer.add_summary(log, i)
  i += 1

  if i % 500 == 0:
    save_name = '%s.ckpt' % model_name
    saver.save(sess, model_path+save_name, global_step=i)

    d = sess.run(dmaps, feed_dict={inflow:img[0:1]})
    d = d[-1].reshape([46,46,52])
    util.vis_dmap(d, 'res_%d.jpg' % i)
    util.vis_dmap(dmap[0], 'truth_%d.jpg' % i)
    misc.imsave('src_%d.jpg' % i, img[0])
    with open('train_log.txt', 'a') as f:
      f.write(names[0])
      f.write('\n')
    



