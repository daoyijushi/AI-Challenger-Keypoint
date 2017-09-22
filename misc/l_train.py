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
r = reader.LReader('./data/train/', 'annotations_new.pkl', 16)

sess = tf.Session()

in_img, in_dmap, in_kmap, pred = network.l1()
labels = tf.placeholder(tf.float32, (None))
loss = tf.losses.log_loss(labels, pred)
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
  img, kmap, dmap, names, truth, link_kmaps = r.next_batch()
  fd = {
    in_img: img,
    in_dmap: dmap,
    labels: truth,
    in_kmap: link_kmaps
  }

  _, batch_loss, step_cnt, log = \
    sess.run([train_step, loss, one_step_op, merged], feed_dict=fd)

  toc = time.time()
  interval = (toc - tic) * 1000

  print('Iter %d, loss %g, timecost %g ms (positive labels %d)' % (step_cnt, batch_loss, interval, np.sum(truth)))
  writer.add_summary(log, step_cnt)

  if step_cnt % 500 == 0 or step_cnt == 1:
    save_name = '%s.ckpt' % model_name
    saver.save(sess, model_path+save_name, global_step=step_cnt)
    



