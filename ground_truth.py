import os
import numpy as np
from scipy import misc
import json

data_dir = './data/'
img_dir = data_dir + 'img/'
keypoint_dir = data_dir + 'keypoint/'
affinity_dir = data_dir + 'affinity/'

# normal distribution, set sigma = sqrt(8)
# that is to make the keypoint is 32x32 large
def normal(x, y):
  return np.exp(-(x**2+y**2)/16) / 16 / np.pi


def normal_patch(sz=(32,32)):
  patch = np.zeros(sz)
  for i in range(sz[0]):
    for j in range(sz[1]):
      patch[i, j] = normal(i-sz[0]/2, j-sz[0]/2)
  return patch


# get the keypoint ground truth
# the first 14 channels are keypoints' locations
def key_map(img, channels=14):
  src = misc.imread(img['image_id'] + '.jpg')
  # src = misc.imread(img_dir + img['image_id'] + '.jpg')
  x, y, _ = src.shape
  print(x, y)
  key_map = np.zeros((x, y, channels + 3))
  patch = normal_patch()
  for keypoints in img['keypoint_annotations'].values():
    for i in range(channels):
      num = 3 * i
      kx = keypoints[num]
      ky = keypoints[num + 1]
      kv = keypoints[num + 2]
      if kv == 1:
        left = max(kx-16, 0)
        right = min(kx+16, x)
        top = max(ky-16, 0)
        down = min(ky+16, y)
  
        key_map[top:down, left:right, i] += \
          patch[16-(kx-left):16+(right-kx), 16-(ky-top):16+(down-ky)]
  return key_map




