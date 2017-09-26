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
  l = get_limbs()
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

  a = np.arange(h, dtype=np.int16).reshape((h, 1))
  grid_h = np.arange(h, dtype=np.int16).reshape((h, 1))
  for i in range(w - 1):
    grid_h = np.concatenate((grid_h,a), axis=1)

  a = np.arange(w, dtype=np.int16).reshape((1,w))
  grid_w = np.arange(w, dtype=np.int16).reshape((1,w))
  for i in range(h - 1):
    grid_w = np.concatenate((grid_w,a), axis=0)

  return grid_h, grid_w

def get_kmap_from_dmap(dmap, limbs, channels=14):
  h, w, _ = dmap.shape
  kmap = np.zeros((h,w,channels))
  # cnt = [0] * channels
  for i,limb in enumerate(limbs):
    start = limb[0]
    end = limb[1]
    # cnt[start] += 1
    # cnt[end] += 1

    # forward
    dx = dmap[:,:,2*i]
    dy = dmap[:,:,2*i+1]
    # kmap[:,:,start] += np.sqrt(np.square(dx)+np.square(dy))
    kmap[:,:,start] = np.maximum(kmap[:,:,start], np.sqrt(np.square(dx)+np.square(dy)))

    if dmap.shape[2] > 26:
      # backward
      dx = dmap[:,:,2*i+26]
      dy = dmap[:,:,2*i+27]
      # kmap[:,:,end] += np.sqrt(np.square(dx)+np.square(dy))
      kmap[:,:,end] = np.maximum(kmap[:,:,end], np.sqrt(np.square(dx)+np.square(dy)))

  # kmap /= cnt
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

def rect_resize(src, target_h, target_w, left=0, top=0):
  h, w, _ = src.shape
  target_rate = target_w / target_h
  src_rate = w / h

  if target_rate < src_rate:
    rate = target_h / h
    tmp = misc.imresize(src, (target_h, int(rate*w)))
    if tmp.shape[1] > target_w:
      left = np.random.randint(0, tmp.shape[1] - target_w)
      right = left + target_w
      tmp = tmp[:, left:right, :]
    else:
      tmp = misc.imresize(tmp, (target_h, target_w))
  else:
    rate = target_w / w
    tmp = misc.imresize(src, (int(rate*h), target_w))
    if tmp.shape[0] > target_h:
      top = np.random.randint(0, tmp.shape[0] - target_h)
      bottom = top + target_h
      tmp = tmp[top:bottom, :, :]
    else:
      tmp = misc.imresize(tmp, (target_h, target_w))

  return tmp, rate, left, top

def rand_resize(src, length, random_flip=True, max_rate=2, delta_px=10):
  h, w, _ = src.shape
  short = min(h, w)
  min_rate = (length + delta_px) / short
  max_rate = min(max_rate, length*2/short)
  rate = np.random.uniform(min_rate, max_rate)
  tmp = misc.imresize(src, rate)

  h, w, _ = tmp.shape
  left = np.random.randint(0, w - length)
  top = np.random.randint(0, h - length)

  tmp = tmp[top:top+length, left:left+length]

  flip = False
  if random_flip:
    if np.random.random() < 0.5:
      tmp = tmp[:,::-1]
      flip = True


  return tmp, rate, left, top, flip

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
      tops.append(0)
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
      tops.append(0)
  imgs = np.array(imgs, dtype=np.float32)
  #lefts = np.array(lefts, dtype=np.uint8)
  #tops = np.array(tops, dtype=np.uint8)
  return imgs, lefts, tops, rate

def multi_rect(src, target_h, target_w, inter_px, left=0, top=0):
  h, w, _ = src.shape
  target_rate = target_w / target_h
  src_rate = w / h
  imgs = []
  lefts = []
  tops = []
  if target_rate < src_rate:
    rate = target_h / h
    tmp = misc.imresize(src, (target_h, int(rate*w)))
    if tmp.shape[1] > target_w:
      for left in range(0, tmp.shape[1] - target_w, inter_px):
        right = left + target_w
        imgs.append(tmp[:, left:right, :])
        lefts.append(left)
        tops.append(0)
      left = tmp.shape[1] - target_w
      right = left + target_w
      imgs.append(tmp[:, left:right, :])
      lefts.append(left)
      tops.append(0)
    else:
      imgs.append(misc.imresize(tmp, (target_h, target_w)))
      lefts.append(0)
      tops.append(0)
  else:
    rate = target_w / w
    tmp = misc.imresize(src, (int(rate*h), target_w))
    if tmp.shape[0] > target_h:
      for top in range(0, tmp.shape[0] - target_h, inter_px):
        bottom = top + target_h
        imgs.append(tmp[top:bottom, :, :])
        lefts.append(0)
        tops.append(top)
      top = tmp.shape[0] - target_h
      bottom = top + target_h
      imgs.append(tmp[top:bottom, :, :])
      lefts.append(0)
      tops.append(top)
    else:
      imgs.append(misc.imresize(tmp, (target_h, target_w)))
      lefts.append(0)
      tops.append(0)
  imgs = np.array(imgs, dtype=np.float32)

  return tmp, rate, left, top

def slide_window(src, length, rate, pic_num=4):
  tmp = misc.imresize(src, rate)
  h, w, _ = tmp.shape
  imgs = []
  lefts = []
  tops = []
  for top in np.linspace(0, h-length, pic_num):
    top = int(top)
    for left in np.linspace(0, w-length, pic_num):
      left = int(left)
      imgs.append(tmp[top:top+length, left:left+length, :])
      lefts.append(lefts)
      tops.append(tops)
  imgs = np.array(imgs, dtype=np.float32)
  return imgs, lefts, tops

#grid: [[0,1,2,3,4,5,...],[0,1,2,3,4,5,...],...]
def find_another(k_slice, d_slice_x, d_slice_y, start_x, start_y, v_x, v_y, grid_h, grid_w):
  mod_v = np.sqrt(v_x**2+v_y**2) + 1e-8

  x_forward = (grid_w - start_x).astype(np.float32)
  y_forward = (grid_h - start_y).astype(np.float32)
  mod_forward = np.sqrt(np.square(x_forward) + np.square(y_forward)) + 1e-8
  
  cos_forward = (x_forward*v_x + y_forward*v_y) / (mod_forward * mod_v)

  mod_backward = np.sqrt(np.square(d_slice_x) + np.square(d_slice_y)).astype(np.float32) + 1e-8
  cos_backward = (d_slice_x*v_x + d_slice_y*v_y) / (mod_backward * mod_v)


  final_map = k_slice * np.exp(cos_forward) * np.exp(cos_backward)

  # vis_slice(final_map, 'final_map.jpg', (772,950))
  # vis_slice(cos_forward, 'cos_forward.jpg', (772,950))
  # vis_slice(cos_backward, 'cos_backward.jpg', (772,950))
  # vis_slice(k_slice, 'k_slice.jpg', (772,950))
  # input()

  y, x = np.unravel_index(np.argmax(final_map), final_map.shape)

  # if y == 30 and x == 36:
  #   print(start_x, start_y)
  #   print(1, final_map[y,x], final_map[9,42])
  #   print(2, k_slice[y,x], k_slice[9,42])
  #   print(3, cos_forward[y,x], cos_forward[9,42])
  #   print(4, cos_backward[y,x], cos_backward[9,42])


  return x, y, k_slice[y,x]

def vis_slice(s, save_name, size):
  vis_s = np.round(s * 255)
  vis_s = misc.imresize(vis_s, size)
  misc.imsave(save_name, vis_s)

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

        # if p1 == 12 and p2 == 13:
        #   print(v_x, v_y)
        # print('find another:', p2)
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
  # for p in new_comer:
  #   print(p, np.round(np.array(known[p])*16.7826))
  # print('known:', known.keys())
  explore(dmap, kmap, new_comer, known, connections, r, grid_h, grid_w, stop_thres, limb_num, channels)

def get_square_patch(x, y, w, h, r):
  left = max(x-r, 0)
  right = min(x+r, w)
  top = max(y-r, 0)
  down = min(y+r, h)
  return left, right, top, down

def clean(dmap, annos, patch, connections):
  pass
  # h, w, _ = dmap.shape
  # r = patch.shape[0] // 2
  # for start,choices in enumerate(connections):
  #   for keypoint in choices:
  #     # print(keypoint)
  #     end, sign, channel = keypoint
  #     if sign == -1: # only calculate once
  #       continue
  #     start_x, start_y = annos[start]
  #     end_x, end_y = annos[end]

  #     v_x = end_x - start_x
  #     v_y = end_y - start_y
  #     mod_v = np.sqrt(v_x**2 + v_y**2) + 1e-8
  #     v_x /= mod_v
  #     v_y /= mod_v

  #     weighted_x = v_x * patch
  #     weighted_y = v_y * patch

  #     # forward
  #     left, right, top, down = get_square_patch(start_x, start_y, h, w, r)
  #     # dmap[top:down, left:right, channel*2] = 0 
  #     # dmap[top:down, left:right, channel*2+1] = 0
  #     # dmap[top:down, left:right, channel*2] -= v_x
  #     # dmap[top:down, left:right, channel*2+1] -= v_y
  #     dmap[top:down, left:right, channel*2] += weighted_x[r-(start_y-top):r+(down-start_y), r-(start_x-left):r+(right-start_x)]
  #     dmap[top:down, left:right, channel*2+1] += weighted_x[r-(start_y-top):r+(down-start_y), r-(start_x-left):r+(right-start_x)]

  #     # backward
  #     left, right, top, down = get_square_patch(end_x, end_y, h, w, r)
  #     # dmap[top:down, left:right, channel*2+26] = 0
  #     # dmap[top:down, left:right, channel*2+1+26] = 0
  #     # dmap[top:down, left:right, channel*2+26] += v_x
  #     # dmap[top:down, left:right, channel*2+1+26] += v_y
  #     dmap[top:down, left:right, channel*2+26] -= weighted_x[r-(end_y-top):r+(down-end_y), r-(end_x-left):r+(right-end_x)]
  #     dmap[top:down, left:right, channel*2+1+26] -= weighted_x[r-(end_y-top):r+(down-end_y), r-(end_x-left):r+(right-end_x)]

  r = patch.shape[0] // 2
  h, w, _ = kmap.shape
  for k, v in annos.items():
    x, y = v
    left = max(x-r, 0)
    right = min(x+r, w)
    top = max(y-r, 0)
    down = min(y+r, h)

    # print('removing', k, 'at', v)
    # print(left, top, right, down)

    kmap[top:down, left:right, k] -= \
      patch[r-(y-top):r+(down-y), r-(x-left):r+(right-x)]
  
  # dmap = np.maximum(dmap, 0)

def rebuild(dmap, kmap, connections, r, grid_h, grid_w, patch, rate, limbs, channels=14):
  # each explore find one person's keypoints
  # result contains each person's annotations in the image
  result = []
  stop_thres = np.percentile(kmap, 99, axis=(0,1))
  limb_num = len(limbs)
  while True:
    annos = {}
    explore(dmap, kmap, [], annos, connections, r, grid_h, grid_w, stop_thres, limb_num, channels)
    if len(annos) == 0:
      break
    # vis_kmap(kmap, 'kmap_before.jpg')
    # clean(dmap, annos, patch, connections)
    # kmap = get_kmap_from_dmap(dmap, limbs)
    # vis_kmap(kmap, 'kmap_after.jpg')
    
    # src = misc.imread('ffa97d027dfc2f2fc62692a035535579c5be74e0.jpg')
    # rev_kmap = get_key_hmap(dmap.shape, [annos2list(annos, 1)], get_patch(10,4), 5)
    # vis_kmap(rev_kmap, 'kmap_extract.jpg')
    # cover_key_map(src, rev_kmap)
    # misc.imsave('vis_anno.jpg', src)
    
    # print('clean')
    # input()
    result.append(annos2list(annos, rate))
    break
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

def validate(x, y, v, h, w, strict):
  if strict:
    if x < 0 or x >= w or y < 0 or y >= h or v != 1:
      return False
  else:
    if x < 0 or x >= w or y < 0 or y >= h or v == 3:
      return False
  return True

# img2map: the rescale ratio between img and map
# for example, img is 368x368, and map is 46x46
# then img2map is 8 (368/46=8)
def concat_maps(batch_maps, lefts, tops, img2map):
  length = batch_maps.shape[1]
  depth = batch_maps.shape[3]
  w = lefts[-1] // img2map + length
  h = tops[-1] // img2map + length
  big_map = np.zeros((h,w,depth))
  cnt = np.zeros((h,w))
  for i in range(len(lefts)):
    left = lefts[i] // img2map
    right = left + length
    top = tops[i] // img2map
    bottom = top + length
    big_map[top:bottom, left:right, :] += batch_maps[i, :, :, :]
    cnt[top:bottom, left:right] += 1
  big_map /= np.reshape(cnt, (h,w,1))
  return big_map

# get the keypoint ground truth
def get_key_hmap(shape, annos, patch, r=5, channels=14, strict=False):
  y, x = shape[0], shape[1]
  key_map = np.zeros((y, x, channels))
  for keypoints in annos:
    for i in range(channels):
      num = 3 * i
      kx = int(keypoints[num])
      ky = int(keypoints[num + 1])
      kv = int(keypoints[num + 2])
      if validate(kx, ky, kv, y, x, strict):
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

def get_single_kmap(shape, x, y, r, patch):
  h = shape[0]
  w = shape[1]
  kmap = np.zeros((h, w))
  left, right, top, down = get_square_patch(x, y, w, h, r)
  kmap[top:down, left:right] = patch[r-(y-top):r+(down-y), r-(x-left):r+(right-x)]
  return kmap

def rescale_annos(annos, rate):
  a = annos.astype(np.float32)
  a[:, ::3] = a[:, ::3] * rate
  a[:, 1::3] = a[:, 1::3] * rate
  a = np.round(a).astype(np.int16)
  return a

# get the limb direction
# the patch should be np.ones
def get_dir_hmap(shape, annos, patch, limbs, r):
  y, x = shape[0], shape[1]
  dir_map = np.zeros((y, x, len(limbs) * 2))
  dir_map_re = np.zeros((y, x, len(limbs * 2)))

  cnt = np.ones((y,x,len(limbs)*2), dtype=np.float32)*1e-8

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

          left = max(x1-r, 0)
          right = min(x1+r, x)
          top = max(y1-r, 0)
          down = min(y1+r, y)
          dir_map[top:down, left:right, channel*2:(channel*2+2)] += \
            (patch[r-(y1-top):r+(down-y1), r-(x1-left):r+(right-x1), :] * v)
          cnt[top:down, left:right, channel*2:(channel*2+2)] += 1


          left = max(x2-r, 0)
          right = min(x2+r, x)
          top = max(y2-r, 0)
          down = min(y2+r, y)
          dir_map_re[top:down, left:right, channel*2:(channel*2+2)] -= \
            (patch[r-(y2-top):r+(down-y2), r-(x2-left):r+(right-x2), :] * v)
          cnt[top:down, left:right, channel*2:(channel*2+2)] += 1

  # print(dir_map[3,14,:])
  # print(cnt[3,14,:])
  dir_map /= cnt
  # print(dir_map[3,14,:])
  dir_map_re /= cnt
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

def draw_limb(aff_map, x1, y1, x2, y2, channel, cnt, r=1):
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
  v_x = np.array(diff_x / dis)
  v_y = np.array(diff_y / dis)

  h, w, _ = aff_map.shape
  if abs(diff_x) > abs(diff_y):
    rate = diff_y / diff_x
    for i in range(x1, x2, step_x):
      mid = np.round(y1 + (i - x1) * rate).astype(int)
      top = max(mid - r, 0)
      down = min(mid + r + 1, h)
      for j in range(top, down):
        aff_map[j, i, channel*2] += v_x / (2 ** abs(j-mid))
        aff_map[j, i, channel*2+1] += v_y / (2 ** abs(j-mid))
        cnt[j, i, channel*2] += 1
        cnt[j, i, channel*2+1] += 1
  else:
    rate = diff_x / diff_y
    for i in range(y1, y2, step_y):
      mid = np.round(x1 + (i - y1) * rate).astype(int)
      left = max(mid - r, 0)
      right = min(mid + r + 1, w)
      for j in range(left, right):
        aff_map[i, j, channel*2] += v_x / (2 ** abs(j-mid))
        aff_map[i, j, channel*2+1] += v_y / (2 ** abs(j-mid))
        cnt[i, j, channel*2] += 1
        cnt[i, j, channel*2+1] += 1

# limbs supposed to be
# ((12,13),(13,3),(13,0),(3,4),(4,5),(0,1),(1,2),(13,9), (9,10),(10,11),(13,6),(6,7),(7,8))
def get_aff_hmap(shape, annos, limbs, strict=False):
  h, w, _ = shape
  aff_map = np.zeros((h, w, len(limbs)*2), dtype=np.float32)
  cnt = np.ones((h, w, len(limbs)*2), dtype=np.float32)*1e-8
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
      if validate(x1, y1, v1, h, w, strict) and validate(x2, y2, v2, h, w, strict):
        draw_limb(aff_map, x1, y1, x2, y2, channel, cnt)

  aff_map /= cnt
  return aff_map

def cover_key_map(img, key_map, channel=0):
  tmp = np.amax(key_map, axis=2)
  tmp *= 255
  tmp = np.round(tmp).astype(np.uint8)
  tmp = np.minimum(tmp, 255)
  mask = (tmp > 1e-3)
  img[mask] = 0
  img[:,:,channel] += tmp
  # for i in range(3):
  #   img[:,:,i] = tmp

def cover_aff_map(img, aff_map):
  # print(aff_map.shape)
  x = aff_map[:,:,::2]
  y = aff_map[:,:,1::2]
  tmp = np.round(np.sqrt(np.square(x) + np.square(y))*255).astype(np.uint8)
  # print(tmp.shape)
  tmp = np.max(tmp, axis=2)
  tmp = np.minimum(tmp, 255)
  mask = (tmp > 1e-3)
  img[mask] = 0
  img[:,:,0] = tmp

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
  cover_aff_map(img, aff_map)
  cover_key_map(img, key_map)
  misc.imsave(save_name, img)

def vis_kmap(kmap, save_name):
  k = np.round(np.max(kmap, axis=2) * 255).astype(np.uint8)
  k = np.minimum(k, 255)
  misc.imsave(save_name, k)

def vis_dmap(dmap, save_name):
  k = get_kmap_from_dmap(dmap, get_limbs())
  vis_kmap(k, save_name)

def vis_amap(amap, save_name):
  x = amap[:,:,::2]
  y = amap[:,:,1::2]
  tmp = np.round(np.sqrt(np.square(x) + np.square(y))*255).astype(np.uint8)
  tmp = np.max(tmp, axis=2)
  tmp = np.minimum(tmp, 255)
  misc.imsave(save_name, tmp)

def vis_layer(layer, save_name):
  l = np.maximum(layer, 0)
  l = np.round(l * 255).astype(np.uint8)
  misc.imsave(save_name, l)

if __name__ == '__main__':
  compute_connections()
  # src = misc.imread('./image/00a63555101c6a702afa83c9865e0296c3cafd6f.jpg')
  # imgs, lefts, tops, rate = multi_resize(src, 368, 24)
  # num = len(imgs)
  # batch_dmaps = np.random.rand(num,46,46,52)
  # big = concat_dmaps(batch_dmaps, lefts, tops, 8)
  # print(big.shape)
  # with open('anno_sample.pkl', 'rb') as f:
  #   a = pickle.load(f)
  # print(a[0])

  # dmap = np.load('dmap.npy')
  # h, w, _ = dmap.shape
  # grid_h, grid_w = get_grid(h, w)
  # kmap = get_kmap_from_dmap(dmap, get_limbs())
  # annos = rebuild(dmap, kmap, get_connections(), 2, grid_h, grid_w, get_patch(10, 4), 0.05958549222797927461139896373057, get_limbs())
  # final = format_annos(annos, 'test')
  # print(final)
  # for i in range(len(final['keypoint_annotations'])):
  #   k_rev = get_key_hmap((772, 950), [final['keypoint_annotations']['human%d'%(i+1)]], get_patch(40,32), r=20)
  #   src = misc.imread('ffa97d027dfc2f2fc62692a035535579c5be74e0.jpg')
  #   cover_key_map(src, k_rev)
  #   misc.imsave('vis_anno_%d.jpg'%i, src)

  # with open('inf_out.json', 'w') as f:
  #   j = json.dumps(final)
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



