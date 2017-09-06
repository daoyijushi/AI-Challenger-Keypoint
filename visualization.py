import numpy as np
from scipy import misc


def visualization(img_path, hmap, save_name='vis.jpg'):
  hmap = np.sum(hmap, axis=2)
  hmap = np.round(hmap * 256)
  maxi = np.max(hmap)
  hmap *= (256 / maxi)
  hmap = hmap.astype(np.uint8)

  img = misc.imread(img_path)
  mask = (hmap != 0)
  img[mask] = 0
  img[:, :, 0] += hmap

  misc.imsave(save_name, img)