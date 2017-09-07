import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

def get_normal_patch(npy_path = './patch.npy'):
  npy_patch = np.load(npy_patch)
  tf_patch = tf.constant(npy_patch, name='normal_patch')
  return tf_patch

def keypoint_heatmap(img, info, patch, channels=14):
  y, x, _ = img.get_shape()
  key_map = tf.zeros((y, x, channels))


