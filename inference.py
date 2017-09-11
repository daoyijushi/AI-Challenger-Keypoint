import network
import tensorflow as tf
import time
import gflags
import sys
import os
import scipy.misc as misc

Flags = gflags.FLAGS

gflags.DEFINE_string('model_path', '', 'where to restore model')
gflags.DEFINE_string('test_path', '', 'where are the test images')
gflags.DEFINE_string('save_path', '', 'where to save the results')

Flags(sys.argv)

model_path = Flags.model_path
test_path = Flags.test_path
save_path = Flags.save_path

names = os.listdir(test_path)

inflow, dmaps = network.dirmap()

sess = tf.Session
saver = tf.train.Saver()

ckpt = tf.train.get_checkpoint_state(model_path)
if ckpt:
  saver.restore(sess, ckpt.model_checkpoint_path)
else:
  print('No available ckpt.')

for name in names:
  img = []
  src = misc.imread(test_path+name)
  # do some resize, all from one same picture
  batch_result = sess.run(dmaps, feed_dict={inflow:img})
  # do some summarize

