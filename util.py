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

def normal(x, y, sigma=4):
  return np.exp(-(x**2+y**2)/sigma)

def normal_patch(l, s):
  patch = np.zeros((l,l))
  for i in range(l):
    for j in range(l):
      patch[i, j] = normal(i-l/2, j-l/2, s)
  return patch

def limbs():
  # 13 limbs
  return ((12,13),(13,3),(13,0),(3,4),(4,5),(0,1),(1,2),(13,9), \
    (9,10),(10,11),(13,6),(6,7),(7,8))

def compute_connections():
  # how are the keypoints connected and the direction
  l = limbs()
  co = []
  for i in range(14):
    connection = []
    for limb in l:
      if limb[0] == i:
        connection.append((limb[1],1))
      elif limb[1] == i:
        connection.append((limb[0],-1))
    co.append(connection)
  print(co)

def connections():
  co = [
    [(13, -1), (1, 1)], #0
    [(0, -1), (2, 1)], #1
    [(1, -1)], #2
    [(13, -1), (4, 1)], #3
    [(3, -1), (5, 1)], #4
    [(4, -1)], #5
    [(13, -1), (7, 1)], #6
    [(6, -1), (8, 1)], #7
    [(7, -1)], #8
    [(13, -1), (10, 1)], #9
    [(9, -1), (11, 1)], #10
    [(10, -1)], #11
    [(13, 1)], #12
    [(12, -1), (3, 1), (0, 1), (9, 1), (6, 1)] #13
  ]
  return co

def get_kmap(dmap, limbs, channels=14):
  h, w, _ = dmap.shape
  kmap = np.zeros((h,w,channels))
  cnt = [0] * channels
  for i,limb in enumerate(limbs):
    start = limb[0]
    end = limb[1]
    cnt[start] += 1
    cnt[end] += 1

    # forward
    dx = dmap[:,:,2*i]
    dy = dmap[:,:,2*i+1]
    kmap[:,:,start] += np.sqrt(np.square(dx)+np.square(dy))

    # backward
    dx = dmap[:,:,2*i+26]
    dy = dmap[:,:,2*i+27]
    kmap[:,:,end] += np.sqrt(np.square(dx)+np.square(dy))

  kmap /= cnt
  return kmap

def explore(dmap, kmap, start, known, connections, channels=14):
  '''
  start: the keypoint(s) from which we explore others, list
  known: the keypoint(s) we've already located, dic
  connections: how are the keypoints connected and the direction, list
  '''
  if len(known == 14):
    return
  if len(start) == 0:
    p = np.argmax(kmap).tolist()
    x, y, c = p
    start.append(c)
    known[c] = (x,y)
  else:
    for p1 in start:
      for limb in connections[p]:
        p2, dire = limb
        hmap = kmap[:,:,p2]


  explore(dmap, kmap, start, known, connections, channels)



def validate(x, y, v, h, w):
  if x < 0 or x >= w or y < 0 or y >= h or v == 3:
    return False
  return True

# get the keypoint ground truth
def get_key_hmap(shape, annos, patch, r, channels=14):
  y, x, _ = shape
  key_map = np.zeros((y, x, channels))
  for keypoints in annos:
    for i in range(channels):
      num = 3 * i
      kx = int(keypoints[num])
      ky = int(keypoints[num + 1])
      kv = int(keypoints[num + 2])
      if validate(kx, ky, kv, y, x):
        left = max(kx-r, 0)
        right = min(kx+r, x)
        top = max(ky-r, 0)
        down = min(ky+r, y)
        # print(kx, ky)
        # print(left, right, top, down)
        # key_map[top:down, left:right, i] = \
        #   np.max(key_map[top:down, left:right, i], patch[r-(ky-top):r+(down-ky), r-(kx-left):r+(right-kx)])
        # key_map[top:down, left:right, i] += \
        #   patch[r-(ky-top):r+(down-ky), r-(kx-left):r+(right-kx)]
        for h in range(top, down):
          for w in range(left, right):
            key_map[h, w, i] = max(key_map[h, w, i], patch[r+h-ky, r+w-kx])
  return key_map

# get the limb direction
# the patch should be np.ones
def get_dir_hmap(shape, annos, patch, limbs, r):
  y, x, _ = shape
  dir_map = np.zeros((y, x, len(limbs) * 2))
  dir_map_re = np.zeros((y, x, len(limbs * 2)))

  cnt = [1e-8, 1e-8] * len(limbs)
  for human in annos:
    for channel, limb in enumerate(limbs):
      num = limb[0] * 3
      x1 = human[num]
      y1 = human[num + 1]
      v1 = human[num + 2]
      num = limb[1] * 3
      x2 = human[num]
      y2 = human[num + 1]
      v2 = human[num + 2]
      if validate(x1, y1, v1, y, x) and validate(x2, y2, v2, y, x):
        diff_x = x2 - x1
        diff_y = y2 - y1
        dis = np.sqrt(diff_x ** 2 + diff_y ** 2)
        if dis != 0:
          v = np.array([diff_x / dis, diff_y / dis])
          cnt[channel * 2] += 1
          cnt[channel * 2 + 1] += 1

          left = max(x1-r, 0)
          right = min(x1+r, x)
          top = max(y1-r, 0)
          down = min(y1+r, y)
          dir_map[top:down, left:right, channel*2:(channel*2+2)] += \
            (patch[r-(y1-top):r+(down-y1), r-(x1-left):r+(right-x1), :] * v)


          left = max(x2-r, 0)
          right = min(x2+r, x)
          top = max(y2-r, 0)
          down = min(y2+r, y)
          dir_map_re[top:down, left:right, channel*2:(channel*2+2)] -= \
            (patch[r-(y2-top):r+(down-y2), r-(x2-left):r+(right-x2), :] * v)

  dir_map /= (np.array(cnt).reshape(len(limbs)*2))
  dir_map_re /= (np.array(cnt).reshape(len(limbs)*2))
  return dir_map, dir_map_re

# add weights for direction map
def weight_dir_hmap(kmap, dmap, dmap_re, limbs):
  for channel,limb in enumerate(limbs):
    start = limb[0]
    weight = kmap[:,:,start]
    dmap[:,:,channel*2] *= weight
    dmap[:,:,channel*2+1] *= weight

    end = limb[1]
    weight = kmap[:,:,end]
    dmap_re[:,:,channel*2] *= weight
    dmap_re[:,:,channel*2+1] *= weight

  dmap = np.concatenate((dmap, dmap_re), axis=2)
  return dmap

def draw_limb(aff_map, x1, y1, x2, y2, channel, r=1):
  x1 = int(x1)
  x2 = int(x2)
  y1 = int(y1)
  y2 = int(y2)
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

# limbs supposed to be
# ((13,14),(14,4),(14,1),(4,5),(5,6),(1,2),(2,3),(14,10),(10,11),(11,12),(14,7),(7,8),(8,9))
def get_aff_hmap(shape, annos, limbs):
  h, w, _ = shape
  aff_map = np.zeros((h, w, len(limbs), 2))
  cnt = [1e-8] * len(limbs)
  for human in annos:
    for channel, limb in enumerate(limbs):
      num = limb[0] * 3
      x1 = human[num]
      y1 = human[num + 1]
      v1 = human[num + 2]
      num = limb[1] * 3
      x2 = human[num]
      y2 = human[num + 1]
      v2 = human[num + 2]
      if validate(x1, y1, v1, h, w) and validate(x2, y2, v2, h, w):
        draw_limb(aff_map, x1, y1, x2, y2, channel)
        cnt[channel] += 1
  aff_map /= (np.array(cnt).reshape(len(limbs),1))
  aff_map = aff_map.reshape((h, w, len(limbs) * 2))
  return aff_map

def cover_key_map(img, key_map):
  key_map = np.amax(key_map, axis=2)
  key_map *= 256
  key_map = np.round(key_map).astype(np.uint8)
  mask = (key_map != 0)
  img[mask] = 0
  img[:,:,0] += key_map
  return img

def cover_aff_map(img, aff_map):
  aff_map = np.sum(aff_map, axis=2) / 14
  aff_map *= 256
  aff_map = np.round(np.abs(aff_map)).astype(np.uint8)
  mask = (aff_map != 0)
  img[mask] = 0
  img[:,:,0] += aff_map
  return img

def test_aff_hmap():
  with open('anno_sample.pickle', 'rb') as f:
    data = pickle.load(f)
  piece = data[1]
  img = misc.imread('./image/' + piece['image_id'] + '.jpg')
  annos = piece['keypoint_annotations'].values()
  aff_map = get_aff_hmap(img.shape, annos, limbs())
  img = cover_aff_map(img, aff_map)
  misc.imsave('aff_map.jpg', img)

def test_key_hmap():
  with open('anno_sample.pkl', 'rb') as f:
    data = pickle.load(f)
  piece = data[1]
  img = misc.imread('./image/' + piece['image_id'] + '.jpg')
  annos = piece['keypoint_annotations'].values()
  key_map = get_key_hmap(img.shape, annos, normal_patch())
  img = cover_key_map(img, key_map)
  misc.imsave('key_map_img.jpg', img)
  key_map = np.amax(key_map, axis=2)
  key_map *= 256
  key_map = np.round(key_map).astype(np.uint8)
  misc.imsave('key_map.jpg', key_map)

def visualization(img, key_map, aff_map, save_name='vis.jpg'):
  img = cover_aff_map(img, aff_map)
  img = cover_key_map(img, key_map)
  misc.imsave(save_name, img)

def vis_dmap(dmap, save_name):
    dx = dmap[:,:,::2]
    dy = dmap[:,:,1::2]
    d = np.sqrt(np.square(dx) + np.square(dy))
    d = np.max(d, axis=2)
    misc.imsave(save_name, d)

def resize(src, anno, length):
  h, w, _ = src.shape
  anno = np.array(anno, dtype=np.float32)
  if h < w:
    rate = length / h
    tmp = misc.imresize(src, (length, int(rate*w)))
    anno[:, ::3] *= rate
    anno[:, 1::3] *= rate
    if tmp.shape[1] > length:
      left = np.random.randint(0, tmp.shape[1] - length)
      right = left + length
      tmp = tmp[:, left:right, :]
      anno[:, ::3] -= left
    else:
      tmp = misc.imresize(src, (length, length))
  else:
    rate = length / w
    tmp = misc.imresize(src, (int(rate*h), length))
    anno[:, ::3] *= rate
    anno[:, 1::3] *= rate
    if tmp.shape[0] > length:
      top = np.random.randint(0, tmp.shape[0] - length)
      bottom = top + length
      tmp = tmp[top:bottom, :, :]
      anno[:, 1::3] -= top
    else:
      tmp = misc.imresize(src, (length, length))
  return tmp, anno.astype(np.int16)

if __name__ == '__main__':
  compute_connections()
  # out = np.load('output.npy')
  # print(out.shape)
  # kmap = get_kmap(out, util.limbs())
  # print(kmap.shape)
  # with open('store.txt', 'w') as f:
  #   content = np.round(kmap[:,:,0], 2).tolist()
  #   for line in content:
  #     for num in line:
  #       f.write(str(num))
  #       f.write(' ')
  #     f.write('\n')



