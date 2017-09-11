import numpy as np
import util
import scipy.misc as misc

def get_kmap(dmap, limbs, channels=14):
  h, w, _ = dmap.shape
  kmap = np.zeros((h,w,channels))
  cnt = [0] * channels
  for i,limb in enumerate(limbs):
    start = limb[0] - 1
    end = limb[1] - 1
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



if __name__ == '__main__':
  out = np.load('output.npy')
  print(out.shape)
  kmap = get_kmap(out, util.limbs())
  print(kmap.shape)
  with open('store.txt', 'w') as f:
    content = np.round(kmap[:,:,0], 2).tolist()
    for line in content:
      for num in line:
        f.write(str(num))
        f.write(' ')
      f.write('\n')



