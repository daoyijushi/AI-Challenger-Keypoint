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

def get_patch(l, s):
  patch = np.zeros((l,l))
  for i in range(l):
    for j in range(l):
      patch[i, j] = normal(i-l/2, j-l/2, s)
  return patch

def get_limbs():
  # 13 limbs
  return ((12,13),(13,3),(13,0),(3,4),(4,5),(0,1),(1,2),(13,9), \
    (9,10),(10,11),(13,6),(6,7),(7,8))

def compute_connections():
  # how are the keypoints connected and the direction
  l = limbs()
  co = []
  for i in range(14):
    connection = []
    for j,limb in enumerate(l):
      if limb[0] == i:
        connection.append((limb[1],1,j))
      elif limb[1] == i:
        connection.append((limb[0],-1,j))
    co.append(connection)
  print(co)

def get_connections():
  co = [
    [(13, -1, 2), (1, 1, 5)], #0
    [(0, -1, 5), (2, 1, 6)], #1
    [(1, -1, 6)], #2
    [(13, -1, 1), (4, 1, 3)], #3
    [(3, -1, 3), (5, 1, 4)], #4
    [(4, -1, 4)], #5
    [(13, -1, 10), (7, 1, 11)], #6
    [(6, -1, 11), (8, 1, 12)], #7
    [(7, -1, 12)], #8
    [(13, -1, 7), (10, 1, 8)], #9
    [(9, -1, 8), (11, 1, 9)], #10
    [(10, -1, 9)], #11
    [(13, 1, 0)], #12
    [(12, -1, 0), (3, 1, 1), (0, 1, 2), (9, 1, 7), (6, 1, 10)] #13
  ]
  return co

# the grid should be the same size of kmap
# coord = 'h' or 'w'
# if h, it should be [[0,0,...],[1,1,...]]
# if w, it should be [[0,1,...],[0,1,...]]
def get_grid(h, w):

  a = np.arrange(0, h, dtype=np.int16).reshape((h, 1))
  grid_h = np.arrange(0, h, dtype=np.int16).reshape((h, 1))
  for i in range(w - 1):
    grid_h = np.concatenate((grid_h,a), axis=1)

  a = np.arrange(0, w, dtype=np.int16).reshape((1,w))
  grid_w = np.arrange(0, w, dtype=np.int16).reshape((1,w))
  for i in range(h - 1):
    grid_w = np.concatenate((grid_w,a), axis=0)

  return grid_h, grid_w

def get_kmap_from_dmap(dmap, limbs, channels=14):
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

def resize(src, length, left=0, top=0):
  h, w, _ = src.shape
  if h < w:
    rate = length / h
    tmp = misc.imresize(src, (length, int(rate*w)))
    if tmp.shape[1] > length:
      left = np.random.randint(0, tmp.shape[1] - length)
      right = left + length
      tmp = tmp[:, left:right, :]
    else:
      tmp = misc.imresize(tmp, (length, length))
  else:
    rate = length / w
    tmp = misc.imresize(src, (int(rate*h), length))
    if tmp.shape[0] > length:
      top = np.random.randint(0, tmp.shape[0] - length)
      bottom = top + length
      tmp = tmp[top:bottom, :, :]
    else:
      tmp = misc.imresize(tmp, (length, length))

  return tmp, rate, left, top

def multi_resize(src, length, inter_px):
  h, w, _ = src.shape
  rate = None
  tmp = None
  imgs = []
  lefts = []
  tops = []
  if h < w:
    rate = length / h
    tmp = misc.imresize(src, (length, int(rate*w)))
    if tmp.shape[1] > length:
      for left in range(0, tmp.shape[1] - length, inter_px):
        right = left + length
        imgs.append(tmp[:, left:right, :])
        lefts.append(left)
        tops.append(0)
      left = tmp.shape[1] - length
      right = left + length
      imgs.append(tmp[:, left:right, :])
      lefts.append(left)
      tops.append(0)
    else:
      imgs.append(misc.imresize(tmp, (length, length)))
      lefts.append(0)
      rights.append(0)
  else:
    rate = length / w
    tmp = misc.imresize(src, (int(rate*h), length))
    if tmp.shape[0] > length:
      for top in range(0, tmp.shape[0] - length, inter_px):
        bottom = top + length
        imgs.append(tmp[top:bottom, :, :])
        lefts.append(0)
        tops.append(top)
      top = tmp.shape[0] - length
      bottom = top + length
      imgs.append(tmp[top:bottom, :, :])
      lefts.append(0)
      tops.append(top)
    else:
      imgs.append(misc.imresize(tmp, (length, length)))
      lefts.append(0)
      rights.append(0)
  return imgs, lefts, tops, rate

#grid: [[0,1,2,3,4,5,...],[0,1,2,3,4,5,...],...]
def find_another(k_slice, d_slice_x, d_slice_y, start_x, start_y, v_x, v_y, grid_h, grid_w):
  mod_v = v_x ** 2 + v_y ** 2

  x_forward = (start_x - grid_w).astype(np.float32)
  y_forward = (start_y - grid_h).astype(np.float32)
  mod_forward = np.sqrt(np.square(x_forward) + np.sqrt(y_forward)) + 1e-4
  
  cos_forward = (x_forward*v_x + y_forward*v_y) / (mod_forward * mod_v)

  mod_backward = np.sqrt(np.square(d_slice_x) + np.square(d_slice_y)).astype(np.float32) + 1e-4
  cos_backward = -(d_slice_x*v_x + d_slice_y*v_y) / (mod_backward * mod_v)

  final_map = k_slice * cos_forward * cos_backward

  y, x = np.unravel_index(np.argmax(k_slice), k_slice.shape)
  return x, y, k_slice[y,x]

def explore(dmap, kmap, start, known, connections, r, grid_h, grid_w, stop_thres, limb_num=13, channels=14):
  '''
  start: the keypoint(s) from which we explore others, list
  known: the keypoint(s) we've already located, dic
  connections: how are the keypoints connected and the direction, list
  '''
  new_comer = []
  if len(known) == 14:
    # print('return')
    return
  if len(start) == 0:
    p = np.unravel_index(np.argmax(kmap), kmap.shape)
    y, x, c = p
    if kmap[y,x,c] < stop_thres[c]:
      return
    new_comer.append(c)
    known[c] = (x,y)
  else:
    h, w, _ = dmap.shape
    for p1 in start:
      p1_x, p1_y = known[p1]
      
      left = max(p1_x-r, 0)
      right = min(p1_x+r, w)
      top = max(p1_y-r, 0)
      down = min(p1_y+r, h)

      # w_slice = kmap[top:down, left:right, p1]
      # w_slice /= np.max(w_slice)
      
      for limb in connections[p1]:
        p2, sign, c = limb

        if p2 in known.keys():
          continue

        k_slice = kmap[:,:,p2]

        d_slice_x = None
        d_slice_y = None
        d_slice_rev_x = None
        d_slice_rev_y = None

        if sign == 1:
          d_slice_x = dmap[top:down, left:right, c*2]
          d_slice_y = dmap[top:down, left:right, c*2+1]
          # find the dmap from p2 to p1
          c_rev = None
          for k in connections[p2]:
            if p1 == k[0]:
              c_rev = k[2]
          d_slice_rev_x = dmap[:,:,c_rev*2+limb_num*2]
          d_slice_rev_y = dmap[:,:,c_rev*2+limb_num*2+1]
        else:
          d_slice_x = dmap[top:down, left:right, c*2+limb_num*2]
          d_slice_y = dmap[top:down, left:right, c*2+limb_num*2+1]
          # find the dmap from p2 to p1
          c_rev = None
          for k in connections[p2]:
            if p1 == k[0]:
              c_rev = k[2]
          d_slice_rev_x = dmap[:,:,c_rev*2]
          d_slice_rev_y = dmap[:,:,c_rev*2+1]

        # vx = np.mean(d_slice_x * w_slice)
        # vy = np.mean(d_slice_y * w_slice)
        v_x = np.mean(d_slice_x)
        v_y = np.mean(d_slice_y)

        p2_x, p2_y, belief = find_another(k_slice,
                                          d_slice_rev_x,
                                          d_slice_rev_y,
                                          p1_x, p1_y,
                                          v_x, v_y,
                                          grid_h, grid_w)

        # if belief > np.mean(k_slice):
        new_comer.append(p2)
        known[p2] = (p2_x,p2_y)

  # print('from', start, 'find', new_comer)
  # print('known:', known.keys())
  explore(dmap, kmap, new_comer, known, connections, r, grid, stop_thres, limb_num, channels)

def clean(kmap, annos, patch):
  r = patch.shape[0] // 2
  h, w, _ = kmap.shape
  for k, v in annos.items():
    x, y = v
    left = max(x-r, 0)
    right = min(x+r, w)
    top = max(y-r, 0)
    down = min(y+r, h)
    kmap[top:down, left:right, k] -= \
      patch[r-(y-top):r+(down-y), r-(x-left):r+(right-x)]
    kmap[kmap < 0] = 0
    # kmap[top:down, left:right, k] = 0

def rebuild(dmap, kmap, connections, r, grid_h, grid_w, patch, rate, limb_num=13, channels=14):
  # each explore find one person's keypoints
  # result contains each person's annotations in the image
  result = []
  stop_thres = np.percentile(kmap, 99, axis=(0,1))
  while True:
    annos = {}
    explore(dmap, kmap, [], annos, connections, 2, grid_h, grid_w, stop_thres, limb_num, channels)
    if len(annos) == 0:
      break
    clean(kmap, annos, patch)
    result.append(annos2list(annos, rate))
  return result

def annos2list(annos, rate):
  l = []
  for k in range(14):
    if k in annos.keys():
      l.append(int(round(annos[k][0] / rate)))
      l.append(int(round(annos[k][1] / rate)))
      l.append(1)
    else:
      l.append(0)
      l.append(0)
      l.append(0)
  return l

def format_annos(annos, img_id):
  ret = {}
  ret['image_id'] = img_id
  ret['keypoint_annotations'] = {}
  h = 'human%d'
  for i,human in enumerate(annos):
    ret['keypoint_annotations'][h%(i+1)] = human
  return ret

def validate(x, y, v, h, w):
  if x < 0 or x >= w or y < 0 or y >= h or v == 3:
    return False
  return True

# img2dmap: the rescale ratio between img and dmap
# for example, img is 368x368, and dmap is 46x46
# then img2dmap is 8 (368/46=8)
def concat_dmaps(batch_dmaps, lefts, tops, img2dmap):
  length = batch_dmaps.shape[1]
  depth = batch_dmaps.shape[3]
  w = lefts[-1] // img2dmap + length
  h = tops[-1] // img2dmap + length
  dmap = np.zeros((h,w,depth))
  cnt = np.zeros((h,w,depth))
  for i in range(len(lefts)):
    left = lefts[i] // img2dmap
    right = left + length
    top = tops[i] // img2dmap
    bottom = top + length
    dmap[top:bottom, left:right, :] += batch_dmaps[i, :, :, :]
    cnt[top:bottom, left:right, :] += 1
  dmap /= cnt
  return dmap

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
# ((12,13),(13,3),(13,0),(3,4),(4,5),(0,1),(1,2),(13,9), (9,10),(10,11),(13,6),(6,7),(7,8))
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

def vis_kmap(kmap, save_name):
  k = np.round(np.max(kmap, axis=2) * 256)
  misc.imsave(save_name, k)

def vis_dmap(dmap, save_name):
  k = get_kmap_from_dmap(dmap, get_limbs())
  vis_kmap(k, save_name)

if __name__ == '__main__':
  src = misc.imread('./image/00a63555101c6a702afa83c9865e0296c3cafd6f.jpg')
  imgs, lefts, tops, rate = multi_resize(src, 368, 24)
  num = len(imgs)
  batch_dmaps = np.random.rand(num,46,46,52)
  big = concat_dmaps(batch_dmaps, lefts, tops, 8)
  print(big.shape)
  # with open('anno_sample.pkl', 'rb') as f:
  #   a = pickle.load(f)
  # print(a[0])
  # dmap = np.load('output.npy')
  # kmap = get_kmap_from_dmap(dmap, get_limbs())
  # annos = rebuild(dmap, kmap, get_connections(), 2, get_grid(46), get_patch(10,4), 0.06)
  # final = format_annos(annos, 'test')
  # # with open('inf_out.json', 'w') as f:
  # j = json.dumps(final)
  # print(j)

  # for human in result:
  #   for i in range(14):
  #     print(i, human[i*3], human[i*3+1], human[i*3+2])
  # vis_dmap(dmap, 'util_vis.jpg')

  # src = misc.imread('./image/00a63555101c6a702afa83c9865e0296c3cafd6f.jpg')
  # imgs, lefts, tops = multi_resize(src, 368, 20)
  # for i,img in enumerate(imgs):
  #   misc.imsave('crop%d.jpg' % i, img)
  # print(lefts)
  # print(tops)

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



