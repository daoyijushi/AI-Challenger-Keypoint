import numpy as np
import util
import queue
import scipy.misc as misc
import os


def flat(k_slice, x, y, r, patch):
  '''
  remove the response peak in kmap at x, y
  technically, we subtract the avg
  '''
  h, w = k_slice.shape
  left, right, top, down = util.get_square_patch(x, y, w, h, r)
  k_slice[top:down, left:right] -= patch[r-(y-top):r+(down-y), r-(x-left):r+(right-x)]
  # for i in range(1,r):
  #   left, right, top, down = util.get_square_patch(x, y, w, h, i)
  #   # k_slice[top:down, left:right] -= np.mean(k_slice[top:down, left:right])
  #   k_slice[top:down, left:right] = 0


def find_outstander(kmap, mask_r, patch):
  ret = []
  h, w, _ = kmap.shape
  response = np.sum(kmap, axis=(0,1))
  layer = np.argmax(response)

  # print(layer)
  
  # stop_thres = np.percentile(kmap[:,:,layer], 99, axis=(0,1))
  stop_thres = 0.15
  # print(stop_thres)
  while True:
    # util.vis_layer(kmap[:,:,layer], 'vis_outstander.jpg')
    # print('pause')
    # input()

    y, x = np.unravel_index(np.argmax(kmap[:,:,layer]), (h,w))

    # print(y, x)
    # print(kmap[y,x,layer])

    if kmap[y,x,layer] < stop_thres:
      break
    ret.append((x, y))

    # print(x, y)
    # util.vis_kmap(kmap, 'before.jpg')
    
    flat(kmap[:,:,layer], x, y, mask_r, patch)
    kmap = np.maximum(kmap, 0)

    # util.vis_kmap(kmap, 'after.jpg')
    # print('flat')
    # input()
  return ret, layer

def find_mask_info(mask_log, x, y, r, s):
  for i,log in enumerate(mask_log):
    px, py, score, p = log
    if (abs(px - x) < r or abs(py - y) < r) and s > score:
      return i
  return -1

def match(x, y, num, dx_forward, dy_forward, dx_backward, dy_backward, grid_h, grid_w, k_slice, k_slice_masked, mask_r, mask_log, patch):
  '''
  x,y: starter's coor
  num: startes's index in list
  '''
  w, h = k_slice.shape
  # vx, vy = dx_forward[y,x], dy_forward[y,x]
  vx, vy = dx_forward[y,x], dy_forward[y,x]
  # print(vx, vy)
  mod_v = np.sqrt(vx**2+vy**2) + 1e-8

  x_forward = (grid_w - x).astype(np.float32)
  y_forward = (grid_h - y).astype(np.float32)
  mod_forward = np.sqrt(np.square(x_forward) + np.square(y_forward)) + 1e-8
  cos_forward = (x_forward*vx + y_forward*vy) / (mod_forward * mod_v)

  mod_backward = np.sqrt(np.square(dx_backward) + np.square(dy_backward)).astype(np.float32) + 1e-8
  cos_backward = (dx_backward*vx + dy_backward*vy) / (mod_backward * mod_v)

  weight = np.exp(cos_forward) * np.exp(cos_backward)

  final_map = k_slice * weight
  final_map_masked = k_slice_masked * weight

  # print('finding for human %d' % num)
  # util.vis_layer(k_slice, 'k_slice.jpg')
  # util.vis_layer(cos_forward, 'cos_forward.jpg')
  # util.vis_layer(cos_backward, 'cos_backward.jpg')
  # util.vis_layer(final_map, 'final_map.jpg')
  # util.vis_layer(final_map, 'final_map_masked.jpg')
  # input()

  opty, optx = np.unravel_index(np.argmax(final_map), final_map.shape)
  opt = final_map[opty, optx]

  opty_masked, optx_masked = np.unravel_index(np.argmax(final_map_masked), final_map.shape)
  opt_masked = final_map_masked[opty_masked, optx_masked]

  if opt_masked == opt:
    mask_log.append((optx, opty, opt, num))
    flat(k_slice_masked, optx, opty, mask_r, patch)
  else:
    i = find_mask_info(mask_log, optx, opty, mask_r, opt)
    if i == -1: # use sub-optimal
      mask_log.append((optx_masked, opty_masked, opt_masked, num))
      flat(k_slice_masked, optx_masked, opty_masked, mask_r, patch)
    else:
      # kick the old man out
      oldx, oldy, _, oldnum = mask_log[i]
      left, right, top, down = util.get_square_patch(oldx, oldy, w, h, mask_r)
      k_slice_masked[top:down, left:right] = k_slice[top:down, left:right]
      # move the new comer in
      mask_log[i] = (optx, opty, opt, num)
      flat(k_slice_masked, optx, opty, mask_r, patch)
      # find the old man a new place
      match(oldx, oldy, oldnum, dx_forward, dx_backward, dy_forward, dy_backward, grid_h, grid_w, k_slice, k_slice_masked, mask_r, mask_log, patch)

def group_match(starters, dx_forward, dy_forward, dx_backward, dy_backward, grid_h, gird_w, k_slice, mask_r, patch):
  # util.vis_layer(k_slice, 'before.jpg')
  k_slice_masked = k_slice.copy()
  h, w = k_slice_masked.shape
  mask_log = []
  for starter in starters:
    x, y = starter[0]
    num = starter[1]
    match(x, y, num, dx_forward, dy_forward, dx_backward, dy_backward, grid_h, gird_w, k_slice, k_slice_masked, mask_r, mask_log, patch)

    # util.vis_layer(k_slice, 'after.jpg')
    # input()

  k_slice_masked = np.reshape(k_slice_masked, (h,w,1))
  orphans, _ = find_outstander(k_slice_masked, mask_r, patch)
  for o in orphans:
    x, y = o
    mask_log.append((x, y, 0, -1))
  return mask_log

def trans_mask_log(mask_log, layer, humans):
  for log in mask_log:
    x, y, _, num = log
    if num == -1:
      humans.append({})
      humans[-1][layer] = (x, y)
    else:
      humans[num][layer] = (x, y)

def reconstruct(dmap, kmap, mask_r, pl=10, ps=4):
  h, w, _ = dmap.shape
  grid_h, grid_w = util.get_grid(h, w)
  connections = util.get_connections()
  limbs = util.get_limbs()
  patch = util.get_patch(pl, ps)

  humans = [] # store the annotation for each human
  q = queue.Queue() # store the layers to be extended from
  used = [False] * 14 # store wether the layer has been explored
  starters, layer = find_outstander(kmap, mask_r, patch)

  # print(layer)
  # print(starters)
  # print(dmap[8,41,0])
  # print(dmap[8,41,1])
  # print(dmap[8,41,26])
  # print(dmap[8,41,27])


  for p in starters:
    dic = {}
    dic[layer] = p
    humans.append(dic)

  q.put(layer)
  used[layer] = True

  debug_cnt = 0

  while True:
    if q.empty():
      # print('empty')
      break
    layer1 = q.get()
    # print('from', layer1)
    starters = []
    for i,h in enumerate(humans):
      if layer1 in h.keys():
        starters.append((h[layer1], i))
    for piece in connections[layer1]:
      debug_cnt += 1
      # if debug_cnt == 2:
      #   return humans
      layer2, sign, d_layer = piece
      if used[layer2]:
        continue
      used[layer2] = True
      q.put(layer2)
      # print('finding', layer2)
      # print('sign', sign)
      # print('dmap layer', d_layer)
      k_slice = kmap[:,:,layer2]

      # in limbs, the vector is from START to END
      # in connections, sign = -1 means layer2 is START
      # forward is from layer1's view to layer2
      # backward is from layer2's view to layer1
      # d_layer is the layer stores vector from START to END

      # so, if sign = 1, layer1 is START, layer2 is END
      # and, dx_forward is from layer1 to layer2
      # so
      dx_forward = dmap[:,:,d_layer*2]
      dy_forward = dmap[:,:,d_layer*2+1]
      dx_backward = dmap[:,:,d_layer*2+26]
      dy_backward = dmap[:,:,d_layer*2+26+1]

      # else, if sign = -1, layer1 is END, layer2 is START
      # and, dx_forward is from layer1 to layer2
      # so
      if sign == -1:
        dx_forward, dx_backward = dx_backward, dx_forward
        dy_forward, dy_backward = dy_backward, dy_forward

      mask_log = group_match(starters, dx_forward, dy_forward, dx_backward, dy_backward, grid_h, grid_w, k_slice, mask_r, patch)
      trans_mask_log(mask_log, layer2, humans)
      # for h in humans:
      #   print(h)
      # print('\n')
      # print(humans)
      # print('mask_log')
      # print(mask_log)

  return humans

def format(humans, img_id, rate):
  ret = {}
  ret['image_id'] = img_id
  ret['keypoint_annotations'] = {}
  for cnt, h in enumerate(humans):
    if len(h) < 5:
      continue
    tmp = []
    for k in range(14):
      if k in h.keys():
        tmp.append(int(round(h[k][0] / rate)))
        tmp.append(int(round(h[k][1] / rate)))
        tmp.append(1)
      else:
        tmp.append(0)
        tmp.append(0)
        tmp.append(0)
    ret['keypoint_annotations']['human%d'%(cnt+1)] = tmp
  return ret

if __name__ == '__main__':
  dmap = np.load('dmap.npy')
  kmap = util.get_kmap_from_dmap(dmap, util.get_limbs())
  humans = reconstruct(dmap, kmap, 5)
  annos = format(humans, 'sample', 0.05958549222797927461139896373057)
  # print(annos)

  for i in range(len(annos['keypoint_annotations'])):
    k_rev = util.get_key_hmap((772, 950), [annos['keypoint_annotations']['human%d'%(i+1)]], util.get_patch(40,32), r=20)
    src = misc.imread('ffa97d027dfc2f2fc62692a035535579c5be74e0.jpg')
    util.cover_key_map(src, k_rev)
    misc.imsave('vis_anno_%d.jpg'%i, src)
  
  # outstander, layer = find_outstander(kmap, 10)
  # print(layer)
  # print(outstander)


