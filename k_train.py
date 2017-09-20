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
r = reader.KReader('./data/train/', 'annotations_new.pkl', 16)

sess = tf.Session()

inflow, training, pred_kmaps, _ = network.k1()
ref, loss, train_step = network.compute_k_loss(pred_kmaps, l_rate)

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
  img, kmap, names = r.next_batch()

  _, batch_loss, step_cnt, log = \
    sess.run([train_step, loss, one_step_op, merged], feed_dict={inflow:img, ref:kmap, training:True})

  toc = time.time()
  interval = (toc - tic) * 1000

  print('Iter %d, loss %g, timecost %g ms' % (step_cnt, batch_loss, interval))
  writer.add_summary(log, step_cnt)

  if step_cnt % 500 == 0 or step_cnt == 1:
    save_name = '%s.ckpt' % model_name
    saver.save(sess, model_path+save_name, global_step=step_cnt)

    k = sess.run(pred_kmaps, feed_dict={inflow:img[0:1], training:False})
    k = k[-1].reshape([46,46,14])
    util.vis_kmap(k, 'pred_%d.jpg' % step_cnt)
    util.vis_kmap(kmap[0], 'truth_%d.jpg' % step_cnt)
    misc.imsave('src_%d.jpg' % step_cnt, img[0])
    with open('train_log.txt', 'a') as f:
      f.write(str(step_cnt))
      f.write(' ')
      f.write(names[0])
      f.write('\n')
    



