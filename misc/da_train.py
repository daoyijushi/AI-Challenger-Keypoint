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
r = reader.DAReader('./data/train/', 'annotations_new.pkl', 16)

sess = tf.Session()

inflow, dmaps, amaps = network.a3()
d_ref, a_ref, d_loss, a_loss, loss = network.compute_loss(dmaps, amaps, 0.5)
train_step = tf.train.AdagradOptimizer(l_rate).minimize(loss)

tf.summary.scalar('total loss', loss)
tf.summary.scalar('dmap loss', d_loss)
tf.summary.scalar('amap loss', a_loss)
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
  img, direction_hmap, affinity_hmap, names = r.next_batch()
  fd = {
    inflow: img, 
    d_ref: direction_hmap,
    a_ref: affinity_hmap
  }

  _, batch_d_loss, batch_a_loss, batch_loss, step_cnt, log = \
    sess.run([train_step, d_loss, a_loss, loss, one_step_op, merged], feed_dict=fd)

  toc = time.time()
  interval = (toc - tic) * 1000

  print('Iter %d, d loss %g, a loss %g, total loss %g, timecost %g ms' % (step_cnt, batch_d_loss, batch_a_loss, batch_loss, interval))
  writer.add_summary(log, step_cnt)

  if step_cnt % 500 == 0 or step_cnt == 1:
    save_name = '%s.ckpt' % model_name
    saver.save(sess, model_path+save_name, global_step=step_cnt)

    d, a = sess.run([dmaps, amaps], feed_dict={inflow:img[0:1]})
    d = d[-1].reshape((46,46,52))
    a = a[-1].reshape((46,46,26))
    tmp = misc.imresize(img[0], (46,46))

    util.vis_amap(affinity_hmap[0], 'truth_amap_%d.jpg'%step_cnt)
    util.vis_dmap(direction_hmap[0], 'truth_dmap_%d.jpg'%step_cnt)

    util.vis_amap(a, 'pred_amap_%d.jpg'%step_cnt)
    util.vis_dmap(d, 'pred_dmap_%d.jpg'%step_cnt)

    misc.imsave('src_%d.jpg'%step_cnt, img[0])

    with open('train_log.txt', 'a') as f:
      f.write(str(int(step_cnt)))
      f.write(' ')
      f.write(names[0])
      f.write('\n')
    



