import os
import numpy as np
from scipy import misc
import json
import pickle

data_dir = './data/'
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
def get_key_map(img, src, channels=14):
  y, x, _ = src.shape
  key_map = np.zeros((y, x, channels))
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

        # print(left, right, top, down)
  
        key_map[top:down, left:right, i] += \
          patch[16-(kx-left):16+(right-kx), 16-(ky-top):16+(down-ky)]

        # print(key_map)

  return key_map

# output file too big, don't use
def generate_key(data_path, img_dir, npy_dir):
  with open(data_path) as f:
    data = json.load(f)
  for piece in data:
    src = misc.imread(img_dir + piece['image_id'] + '.jpg')
    key_map = get_key_map(piece, src)
    print(src.shape)
    print(key_map.shape)
    result = np.concatenate((key_map, src), axis=2)
    print(result.shape)
    np.save(npy_dir + piece['image_id'] + '.npy', result)

