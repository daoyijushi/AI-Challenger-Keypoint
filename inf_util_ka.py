import numpy as np
import util
import scipy.misc as misc
import os
import queue

def resize_map(small, target_h, target_w):
  depth = small.shape[-1]
  big = np.zeros((target_h, target_w, depth))
  for i in range(depth):
    big[:,:,i] = misc.imresize(small[:,:,i], (target_h, target_w))
  return big

def aff_score(x1, y1, x2, y2, ax, ay, r):
  h, w = ax.shape

  # simply seperate the two points a little bit
  if x1 == x2 and y1 == y2:
    x1, x2, y1, y2 = util.get_square_patch(x1, y1, w, h, 1)

  diff_x = x2 - x1
  diff_y = y2 - y1
  if diff_x == 0 and diff_y == 0:
    print('Error: same point in get_aff_score')

  if diff_x > 0:
    step_x = 1
  else:
    step_x = -1
  if diff_y > 0:
    step_y = 1
  else:
    step_y = -1

  dis = np.sqrt(diff_x**2 + diff_y**2)
  vx = diff_x / dis
  vy = diff_y / dis

  score = 0
  if abs(diff_x) > abs(diff_y):
    rate = diff_y / diff_x
    for i in range(x1, x2, step_x):
      mid = np.round(y1 + (i - x1) * rate).astype(int)
      top = max(mid - r, 0)
      down = min(mid + r, h)
      for j in range(top, down):
        score += ((vx*ax[j,i] + vy*ay[j,i]) / (2**abs(j-mid)))
  else:
    rate = diff_x / diff_y
    for i in range(y1, y2, step_y):
      mid = np.round(x1 + (i - y1) * rate).astype(int)
      left = max(mid - r, 0)
      right = min(mid + r, w)
      for j in range(left, right):
        score += ((vx*ax[i,j] + vy*ay[i,j]) / (2**abs(j-mid)))

  score /= dis
  return score

def flat(k_slice, x, y, r):
  '''
  remove the response peak in kmap at x, y
  technically, we subtract the avg
  '''
  h, w = k_slice.shape
  left, right, top, down = util.get_square_patch(x, y, w, h, r)
  patch = util.get_patch(r*2, 16)
  for i in reversed(range(1,r)):
    left, right, top, down = util.get_square_patch(x, y, w, h, i)
    # k_slice[top:down, left:right] -= np.mean(k_slice[top:down, left:right])
    k_slice[top:down, left:right] -= patch[r-(y-top):r+(down-y), r-(x-left):r+(right-x)]
    # k_slice[top:down, left:right] = 0
  k_slice = np.maximum(k_slice, 0)

def find_outstander_layer(kslice, mask_r, stop_thres, target=0):
  ret = []
  while True:

    # util.vis_layer(kmap[:,:,layer], 'vis_outstander.jpg')
    # print('pause')
    # input()

    y, x = np.unravel_index(np.argmax(kslice), kslice.shape)

    # print(y, x)
    # print(kmap[y,x,layer])

    if kslice[y, x] < stop_thres and len(ret) >= target:
      break
    ret.append((x, y))

    # print(x, y)
    # util.vis_layer(kslice, 'before.jpg')

    flat(kslice, x, y, mask_r)
    kslice = np.maximum(kslice, 0)

    # util.vis_layer(kslice, 'after.jpg')
    # print('flat')
    # input()
  return ret

def find_outstander_brick(kmap, mask_r, stop_thres):
  h, w, _ = kmap.shape
  response = np.sum(kmap, axis=(0,1))
  layer = np.argmax(response)
  ret = find_outstander_layer(kmap[:,:,layer], mask_r, stop_thres)

  return ret, layer

def score_mat(start, end, ax, ay, sign, limb_r):
  h = len(start)
  w = len(end)
  mat = np.zeros((h,w))
  for i, p1 in enumerate(start):
    x1, y1 = p1[0]
    for j, p2 in enumerate(end):
      x2, y2 = p2
      mat[i, j] = sign * aff_score(x1, y1, x2, y2, ax, ay, limb_r)
  return mat

def opt_match(mat):
  h, w = mat.shape
  match = []
  big = -2147483648
  for cnt in range(min(h,w)):
    p1, p2 = np.unravel_index(np.argmax(mat), mat.shape)
    match.append((p1, p2))
    mat[p1,:] = big
    mat[:,p2] = big
  return match

def reconstruct(amap, kmap, mask_r):
  connections = util.get_connections()
  height, width, _ = amap.shape

  # stop_thres = np.percentile(kmap[:,:,layer], 99, axis=(0,1))
  stop_thres = 0.15
  limb_r = 1

  humans = []
  q = queue.Queue()
  used = [False] * 14
  start, layer = find_outstander_brick(kmap, mask_r, stop_thres)

  for p in start:
    dic = {}
    dic[layer] = p
    humans.append(dic)

  q.put(layer)
  used[layer] = True

  while True:
    if q.empty():
      break

    layer1 = q.get()
    start = []
    for i, h in enumerate(humans):
      if layer1 in h.keys():
        start.append((h[layer1], i))

    for piece in connections[layer1]:
      layer2, sign, a_layer = piece

      if used[layer2]:
        continue

      used[layer2] = True
      q.put(layer2)

      # print('finding layer%d' % layer2)
      k_slice = kmap[:,:,layer2]

      # util.vis_layer(k_slice, 'k%d.jpg'%layer2, 'ffa97d027dfc2f2fc62692a035535579c5be74e0.jpg')

      end = find_outstander_layer(k_slice, mask_r, stop_thres, target=len(start))
      # print(end)

      ax = amap[:,:,a_layer*2]
      ay = amap[:,:,a_layer*2+1]

      mat = score_mat(start, end, ax, ay, sign, limb_r)
      match = opt_match(mat)
      for pair in match:
        i1, i2 = pair
        h1 = start[i1][1]
        humans[h1][layer2] = end[i2]

  return humans

def format(humans, img_id, rate):
  ret = {}
  ret['image_id'] = img_id
  ret['keypoint_annotations'] = {}
  ratio = 1 / rate
  base = ratio / 2
  for cnt, h in enumerate(humans):
    if len(h) < 5:
      continue
    tmp = []
    for k in range(14):
      if k in h.keys():
        tmp.append(int(round(h[k][0]*ratio + base)))
        tmp.append(int(round(h[k][1]*ratio + base)))
        tmp.append(1)
      else:
        tmp.append(0)
        tmp.append(0)
        tmp.append(0)
    ret['keypoint_annotations']['human%d'%(cnt+1)] = tmp
  return ret

if __name__ == '__main__':
  amap = np.load('amap.npy')
  kmap = np.load('kmap.npy')

  humans = reconstruct(amap, kmap, 10)
  annos = format(humans, 'ffa97d027dfc2f2fc62692a035535579c5be74e0', 0.05958549222797927461139896373057)
  # annos = format(humans, 'ffa97d027dfc2f2fc62692a035535579c5be74e0', 1)
  print(annos)

  for i in range(len(annos['keypoint_annotations'])):
    k_rev = util.get_key_hmap((772, 950), [annos['keypoint_annotations']['human%d'%(i+1)]], util.get_patch(40,32), r=20)
    # k_slice = k_rev[:,:,8].copy()
    # k_rev[:,:,:] = 0
    # k_rev[:,:,8] = k_slice
    src = misc.imread('ffa97d027dfc2f2fc62692a035535579c5be74e0.jpg')
    util.cover_key_map(src, k_rev)
    misc.imsave('vis_anno_%d.jpg'%i, src)
