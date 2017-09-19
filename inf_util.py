import numpy as np
import util
import Queue



def flat(k_slice, x, y, r):
  '''
  remove the response peak in kmap at x, y
  technically, we subtract the avg
  '''
  h, w = k_slice.shape
  for i in range(1,r):
    left, right, top, down = util.get_square_patch(x, y, w, h, i)
    k_slice[top:down, left:right] -= np.mean(k_slice[top:down, left:right])

def find_outstander(kmap, mask_r):
  ret = []
  h, w, _ = kmap.shape
  response = np.sum(kmap, axis=(0,1))
  layer = np.argmax(response)

  # print(layer)
  
  stop_thres = np.percentile(kmap[:,:,layer], 99, axis=(0,1))
  while True:
    y, x = np.unravel_index(np.argmax(kmap[:,:,layer]), (h,w))
    if kmap[y,x,layer] < stop_thres:
      break
    ret.append((x, y))

    # print(x, y)
    # util.vis_kmap(kmap, 'before.jpg')
    
    flat(kmap[:,:,layer], x, y, mask_r)

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

def match(x, y, num, dx_forward, dy_forward, dx_backward, dy_backward, grid_h, gird_w, k_slice, k_slice_masked, mask_r, mask_log):
  '''
  x,y: starter's coor
  num: startes's index in list
  '''
  w, h = k_slice.shape
  vx, vy = dx_forward[y,x], dy_forward[y,x]
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

  opty, optx = np.unravel_index(np.argmax(final_map), final_map.shape)
  opt = final_map[opty, optx]

  opty_masked, optx_masked = np.unravel_index(np.argmax(final_map_masked), final_map.shape)
  opt_masked = final_map_masked[opty_masked, optx_masked]

  if opt_masked == opt:
    mask_log.append((optx, opty, opt, num))
    flat(k_slice_masked)
  else:
    i = find_mask_info(mask_log, optx, opty, mask_r, opt)
    if i == -1: # use sub-optimal
      mask_log.append((optx_masked, opty_masked, opt_masked, num))
      flat(k_slice_masked)
    else:
      # kick the old man out
      oldx, oldy, _, oldnum = mask_log[i]
      left, right, top, down = util,get_square_patch(oldx, oldy, w, h, mask_r)
      k_slice_masked[top:down, left:right] = k_slice[top:down, left:right]
      # move the new comer in
      mask_log[i] = (optx, opty, opt, num)
      flat(k_slice_masked, optx, opty, mask_r)
      # find the old man a new place
      match(oldx, oldy, oldnum, dx_forward, dx_backward, dy_forward, dy_backward, grid_h, grid_w, k_slice, k_slice_masked, mask_r, mask_log)

def group_match(starters, dx_forward, dy_forward, dx_backward, dy_backward, grid_h, gird_w, k_slice, mask_r):
  k_slice_masked = k_slice.copy()
  h, w = k_slice_masked.shape
  mask_log = []
  for num, starter in enumerate(starters):
    x, y = starter
    match(x, y, num, dx_forward, dy_forward, dx_backward, dy_backward, grid_h, gird_w, k_slice, k_slice_masked, mask_r, mask_log)
  k_slice_masked = np.reshape(k_slice_masked, (h,w,1))
  orphans, _ = find_outstander(k_slice_masked)
  for o in orphans:
    x, y = o
    mask_log.append(x, y, 0, -1)
  return mask_log

def trans_mask_log(mask_log, layer, humans):
  for log in mask_log:
    x, y, _, num = log
    if num == -1:
      humans.append({})
      humans[-1][layer] = (x, y)
    else:
      humans[num][layer] = (x, y)

def reconstruct(dmap, kmap, mask_r):
  h, w, _ = dmap.shape
  grid_h, grid_w = util.get_grid(h, w)
  connections = util.get_connections()
  limbs = util.get_limbs()
  
  humans = [] # store the annotation for each human
  queue = Queue.Queue() # store the layers to be extended from
  used = [False] * 14 # store wether the layer has been explored

  starters, layer = find_outstander(kmap, mask_r)
  for p in starters:
    dic = {}
    dic[layer] = p
    humans.append(dic)

  queue.put(layer)
  used[layer] = True
  while True:
    if queue.empty():
      break

    layer1 = queue.get()

    starters = []
    for h in humans:
      starters.append(h[layer1])
    for piece in connections[layer1]:
      layer2, sign, d_layer = piece
      if used[layer2]:
        break
      used[layer2] = True
      queue.put(layer2)
      k_slice = kmap[:,:,layer2]
      dx_forward = dmap[:,:,d_layer*2]
      dy_forward = dmap[:,:,d_layer*2+1]
      dx_backward = dmap[:,:,d_layer*2+26]
      dy_backward = dmap[:,:,d_layer*2+26+1]
      if sign == -1:
        dx_forward, dx_backward = dx_backward, dx_forward
        dy_forward, dy_backward = dy_backward, dy_forward
      mask_log = group_match(starters, dx_forward, dy_forward, dx_backward, dy_backward, grid_h, grid_w, k_slice, mask_r)
      trans_mask_log(mask_log, humans)



if __name__ == '__main__':
  dmap = np.load('dmap.npy')
  kmap = util.get_kmap_from_dmap(dmap, util.get_limbs())
  outstander, layer = find_outstander(kmap, 10)
  print(layer)
  print(outstander)