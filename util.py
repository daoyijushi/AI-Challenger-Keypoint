import os
import numpy as np
from scipy import misc
import json
import pickle

def json2pickle(fname):
  with open(fname, 'r') as f:
    data = json.load(f)
  with open(fname.split('.')[0]+'.pkl', 'wb') as f:
    pickle.dump(data, f)

# normal distribution, set sigma = 4
# that is to make the keypoint is 16x16 large
def normal(x, y):
  return np.exp(-(x**2+y**2)/8) / 8 / np.pi

def normal_patch(l=16):
  patch = np.zeros((l,l))
  for i in range(l):
    for j in range(l):
      patch[i, j] = normal(i-l/2, j-l/2)
  patch /= patch[l//2,l//2] # make the center = 1
  return patch

def limbs():
  return ((13,14),(14,4),(14,1),(4,5),(5,6),(1,2),(2,3),(14,10), \
    (10,11),(11,12),(14,7),(7,8),(8,9))


def validate(x, y, v, h, w):
  if x < 0 or x >= w or y < 0 or y >= h or v == 3:
    return False
  return True

# get the keypoint ground truth
def get_key_hmap(shape, annos, patch, channels=14, r=8):
  y, x, _ = shape
  key_map = np.zeros((y, x, channels))
  for keypoints in annos:
    for i in range(channels):
      num = 3 * i
      kx = keypoints[num]
      ky = keypoints[num + 1]
      kv = keypoints[num + 2]
      if validate(kx, ky, kv, y, x):
        left = max(kx-r, 0)
        right = min(kx+r, x)
        top = max(ky-r, 0)
        down = min(ky+r, y)
        # print(kx, ky)
        # print(left, right, top, down)
        key_map[top:down, left:right, i] += \
          patch[r-(ky-top):r+(down-ky), r-(kx-left):r+(right-kx)]
  return key_map

def draw_limb(aff_map, x1, y1, x2, y2, channel, r=3):
  diff_x = x2 - x1
  diff_y = y2 - y1
  if diff_x == 0 and diff_y == 0:
    return

  if diff_x > 0:
    step_x = 1
  else:
    step_x = -1
  if diff_y > 0:
    step_y = 1
  else:
    step_y = -1

  dis = np.sqrt(diff_x ** 2 + diff_y ** 2)
  v = np.array([diff_x / dis, diff_y / dis])

  h, w, _, _ = aff_map.shape
  if abs(diff_x) > abs(diff_y):
    rate = diff_y / diff_x
    for i in range(x1, x2, step_x):
      mid = np.round(y1 + (i - x1) * rate).astype(int)
      top = max(mid - r, 0)
      down = min(mid + r, h)
      for j in range(top, down):
        aff_map[j, i, channel, :] += v
  else:
    rate = diff_x / diff_y
    for i in range(y1, y2, step_y):
      mid = np.round(x1 + (i - y1) * rate).astype(int)
      left = max(mid - r, 0)
      right = min(mid + r, w)
      for j in range(left, right):
        aff_map[i, j, channel, :] += v

# limbs is the connection between keypoints
# supposed to be
# ((13,14),(14,4),(14,1),(4,5),(5,6),(1,2),(2,3),(14,10),(10,11),(11,12),(14,7),(7,8),(8,9))
def get_aff_hmap(shape, annos, limbs):
  h, w, _ = shape
  aff_map = np.zeros((h, w, len(limbs), 2))
  cnt = [1e-8] * len(limbs)
  for human in annos:
    for channel, limb in enumerate(limbs):
      num = (limb[0] - 1) * 3
      x1 = human[num]
      y1 = human[num + 1]
      v1 = human[num + 2]
      num = (limb[1] - 1) * 3
      x2 = human[num]
      y2 = human[num + 1]
      v2 = human[num + 2]
      if validate(x1, y1, v1, h, w) and validate(x2, y2, v2, h, w):
        draw_limb(aff_map, x1, y1, x2, y2, channel)
        cnt[channel] += 1
  aff_map /= (np.array(cnt).reshape(len(limbs),1))
  return aff_map

def cover_key_map(img, key_map):
  key_map = np.sum(key_map, axis=2)
  key_map *= 256
  key_map = np.round(key_map).astype(np.uint8)
  mask = (key_map != 0)
  img[mask] = 0
  img[:,:,0] += key_map
  return img

def cover_aff_map(img, aff_map):
  aff_map = np.sum(aff_map, axis=(2,3)) / 14
  aff_map *= 256
  aff_map = np.round(np.abs(aff_map)).astype(np.uint8)
  mask = (aff_map != 0)
  img[mask] = 0
  img[:,:,0] += aff_map
  return img

def test_aff_hmap():
  with open('anno_sample.json', 'r') as f:
    data = json.load(f)
  piece = data[1]
  img = misc.imread('./image/' + piece['image_id'] + '.jpg')
  annos = piece['keypoint_annotations'].values()
  limbs = ((13,14),(14,4),(14,1),(4,5),(5,6),(1,2),(2,3),(14,10),(10,11),(11,12),(14,7),(7,8),(8,9))
  aff_map = get_aff_hmap(img.shape, annos, limbs)
  img = cover_aff_map(img, aff_map)
  misc.imsave('aff_map.jpg', img)

def test_key_hmap():
  with open('anno_sample.json', 'r') as f:
    data = json.load(f)
  piece = data[1]
  img = misc.imread('./image/' + piece['image_id'] + '.jpg')
  annos = piece['keypoint_annotations'].values()
  key_map = get_key_hmap(img.shape, annos, normal_patch())
  img = cover_key_map(img, key_map)
  misc.imsave('key_map.jpg', img)

def visualization(img, key_map, aff_map, save_name='vis.jpg'):
  img = cover_aff_map(img, aff_map)
  img = cover_key_map(img, key_map)
  misc.imsave(save_name, img)

if __name__ == '__main__':
  json2pickle('annotations.json')


