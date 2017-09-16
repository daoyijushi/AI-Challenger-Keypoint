import util
import pickle
import random
from scipy import misc
import numpy as np
import random

class Reader:

  def __init__(self, img_dir, anno_path, batch_size, l1=368, l2=46):
    self.img_dir = img_dir
    with open(anno_path, 'rb') as f:
      self.data = pickle.load(f)
    self.batch_size = batch_size
    self.index = 0
    self.volumn = len(self.data)
    self.length = l1
    self.short = l2
    random.shuffle(self.data)
    self.patch = util.normal_patch(10)
    self.limbs = util.limbs()
    self.mean = np.array([122.35131039, 115.17054545, 107.60200075])
    self.var = np.array([35.77071304, 35.39201422, 37.7260754])
    np.random.seed(822)
    print('Reader initialized. Data volumn %d, batch size %d.' \
      % (self.volumn, self.batch_size))

  def next_batch(self):
    start = self.index
    end = self.index + self.batch_size
    if end > self.volumn:
      end = self.volumn
      self.index = 0
    else:
      self.index = end
    data_batch = self.data[start:end]
    img = []
    keypoint_hmap = []
    affinity_hmap = []
    for piece in data_batch:
      tmp = misc.imread(self.img_dir + piece['image_id'] + '.jpg')
      tmp, annos = util.resize(tmp, list(piece['keypoint_annotations'].values()). self.length)
      img.append(tmp)
      tmp = misc.imresize(tmp, (self.short, self.short))
      annos = np.round(annos * self.short / self.length)
      keypoint_hmap.append(util.get_key_hmap(tmp.shape, annos, self.patch))
      affinity_hmap.append(util.get_aff_hmap(tmp.shape, annos, self.limbs))
    img = np.array(img, dtype=np.float32)
    img -= self.mean
    img /= self.var
    keypoint_hmap = np.array(keypoint_hmap)
    affinity_hmap = np.array(affinity_hmap)
    return img, keypoint_hmap, affinity_hmap

class DirReader:

  def __init__(self, img_dir, anno_path, batch_size, pl=10, ps=4, l1=368, l2=46):
    self.img_dir = img_dir
    with open(anno_path, 'rb') as f:
      self.data = pickle.load(f)
    self.batch_size = batch_size
    self.index = 0
    self.volumn = len(self.data)
    self.length = l1
    self.short = l2
    self.patch_l = pl
    self.patch_s = ps
    random.shuffle(self.data)
    self.patch = util.get_patch(self.patch_l, self.patch_s)
    self.ones = np.ones((self.patch_l,self.patch_l,2))
    self.limbs = util.get_limbs()
    self.mean = np.array([122.35131039, 115.17054545, 107.60200075])
    self.var = np.array([35.77071304, 35.39201422, 37.7260754])
    np.random.seed(822)
    print('DirReader initialized. Data volumn %d, batch size %d.' \
      % (self.volumn, self.batch_size))

  def next_batch(self):
    start = self.index
    end = self.index + self.batch_size
    if end >= self.volumn:
      end = self.volumn
      self.index = 0
    else:
      self.index = end

    data_batch = self.data[start:end]
    img = []
    keypoint_hmap = []
    direction_hmap = []
    for piece in data_batch:
      tmp = misc.imread(self.img_dir + piece['image_id'] + '.jpg')
      
      tmp, rate, left, top = \
        util.resize(tmp, self.length)
      annos = np.array(list(piece['keypoint_annotations'].values()), dtype=np.float16)
      annos[:, ::3] = annos[:, ::3] * rate - left
      annos[:, 1::3] = annos[:, 1::3] * rate - top
      annos = annos.astype(np.int16)

      img.append(tmp)

      tmp = misc.imresize(tmp, (self.short, self.short))
      # annos = np.round(annos * self.short / self.length).astype(np.int8)
      rate = self.short / self.length
      annos[:, ::3] = annos[:, ::3] * rate
      annos[:, 1::3] = annos[:, 1::3] * rate
      annos = np.round(annos).astype(np.int8)
      # print(annos)

      kmap = util.get_key_hmap(\
        tmp.shape, annos, self.patch, self.patch_l//2)
      keypoint_hmap.append(kmap)

      tmp_dmap, tmp_dmap_re = util.get_dir_hmap(\
        tmp.shape, annos, self.ones, self.limbs, self.patch_l//2)

      dmap = util.weight_dir_hmap(kmap, tmp_dmap, tmp_dmap_re, self.limbs)

      direction_hmap.append(dmap)

    img = np.array(img, dtype=np.float32)
    img -= self.mean
    img /= self.var
    keypoint_hmap = np.array(keypoint_hmap, dtype=np.float32)
    direction_hmap = np.array(direction_hmap, dtype=np.float32)
    return img, keypoint_hmap, direction_hmap

class ForwardReader:

  def __init__(self, img_dir, anno_path, batch_size, pl=10, ps=4, l1=368, l2=46):
    self.img_dir = img_dir
    with open(anno_path, 'rb') as f:
      self.data = pickle.load(f)
    self.batch_size = batch_size
    self.index = 0
    self.volumn = len(self.data)
    self.length = l1
    self.short = l2
    self.patch_l = pl
    self.patch_s = ps
    random.shuffle(self.data)
    self.patch = util.get_patch(self.patch_l, self.patch_s)
    self.ones = np.ones((self.patch_l,self.patch_l,2))
    self.limbs = util.get_limbs()
    self.mean = np.array([122.35131039, 115.17054545, 107.60200075])
    self.var = np.array([35.77071304, 35.39201422, 37.7260754])
    np.random.seed(822)
    print('DirReader initialized. Data volumn %d, batch size %d.' \
      % (self.volumn, self.batch_size))

  def next_batch(self):
    start = self.index
    end = self.index + self.batch_size
    if end >= self.volumn:
      end = self.volumn
      self.index = 0
    else:
      self.index = end

    data_batch = self.data[start:end]
    img = []
    keypoint_hmap = []
    direction_hmap = []
    for piece in data_batch:
      tmp = misc.imread(self.img_dir + piece['image_id'] + '.jpg')
      
      tmp, rate, left, top = \
        util.resize(tmp, self.length)
      annos = np.array(list(piece['keypoint_annotations'].values()), dtype=np.float16)
      annos[:, ::3] = annos[:, ::3] * rate - left
      annos[:, 1::3] = annos[:, 1::3] * rate - top
      annos = annos.astype(np.int16)

      img.append(tmp)

      tmp = misc.imresize(tmp, (self.short, self.short))
      # annos = np.round(annos * self.short / self.length).astype(np.int8)
      rate = self.short / self.length
      annos[:, ::3] = annos[:, ::3] * rate
      annos[:, 1::3] = annos[:, 1::3] * rate
      annos = np.round(annos).astype(np.int8)
      # print(annos)

      kmap = util.get_key_hmap(\
        tmp.shape, annos, self.patch, self.patch_l//2)
      keypoint_hmap.append(kmap)

      tmp_dmap, tmp_dmap_re = util.get_dir_hmap(\
        tmp.shape, annos, self.ones, self.limbs, self.patch_l//2)

      dmap = util.weight_dir_hmap(kmap, tmp_dmap, tmp_dmap_re, self.limbs)

      # use forward only
      dmap = dmap[:,:,:len(self.limbs)*2]

      direction_hmap.append(dmap)

    img = np.array(img, dtype=np.float32)
    img -= self.mean
    img /= self.var
    keypoint_hmap = np.array(keypoint_hmap, dtype=np.float32)
    direction_hmap = np.array(direction_hmap, dtype=np.float32)
    return img, keypoint_hmap, direction_hmap

class ForwardReader:

  def __init__(self, img_dir, anno_path, batch_size, pl=10, ps=4, l1=368, l2=46):
    self.img_dir = img_dir
    with open(anno_path, 'rb') as f:
      self.data = pickle.load(f)
    self.batch_size = batch_size
    self.index = 0
    self.volumn = len(self.data)
    self.length = l1
    self.short = l2
    self.patch_l = pl
    self.patch_s = ps
    random.shuffle(self.data)
    self.patch = util.get_patch(self.patch_l, self.patch_s)
    self.ones = np.ones((self.patch_l,self.patch_l,2))
    self.limbs = util.get_limbs()
    self.mean = np.array([122.35131039, 115.17054545, 107.60200075])
    self.var = np.array([35.77071304, 35.39201422, 37.7260754])
    np.random.seed(822)
    print('DirReader initialized. Data volumn %d, batch size %d.' \
      % (self.volumn, self.batch_size))

  def next_batch(self):
    start = self.index
    end = self.index + self.batch_size
    if end >= self.volumn:
      end = self.volumn
      self.index = 0
    else:
      self.index = end

    data_batch = self.data[start:end]
    img = []
    keypoint_hmap = []
    direction_hmap = []
    for piece in data_batch:
      tmp = misc.imread(self.img_dir + piece['image_id'] + '.jpg')
      
      tmp, rate, left, top = \
        util.resize(tmp, self.length)
      annos = np.array(list(piece['keypoint_annotations'].values()), dtype=np.float16)
      annos[:, ::3] = annos[:, ::3] * rate - left
      annos[:, 1::3] = annos[:, 1::3] * rate - top
      annos = annos.astype(np.int16)

      img.append(tmp)

      tmp = misc.imresize(tmp, (self.short, self.short))
      # annos = np.round(annos * self.short / self.length).astype(np.int8)
      rate = self.short / self.length
      annos[:, ::3] = annos[:, ::3] * rate
      annos[:, 1::3] = annos[:, 1::3] * rate
      annos = np.round(annos).astype(np.int8)
      # print(annos)

      kmap = util.get_key_hmap(\
        tmp.shape, annos, self.patch, self.patch_l//2)
      keypoint_hmap.append(kmap)

      tmp_dmap, tmp_dmap_re = util.get_dir_hmap(\
        tmp.shape, annos, self.ones, self.limbs, self.patch_l//2)

      dmap = util.weight_dir_hmap(kmap, tmp_dmap, tmp_dmap_re, self.limbs)

      # use forward only
      dmap = dmap[:,:,len(self.limbs)*2:]
      
      direction_hmap.append(dmap)

    img = np.array(img, dtype=np.float32)
    img -= self.mean
    img /= self.var
    keypoint_hmap = np.array(keypoint_hmap, dtype=np.float32)
    direction_hmap = np.array(direction_hmap, dtype=np.float32)
    return img, keypoint_hmap, direction_hmap

if __name__ == '__main__':
  r = DirReader('./image/', 'anno_sample.pkl', 1)
  i, k, d = r.next_batch()

  k = k[0]
  d = d[0]
  i = i[0]

  k = util.get_kmap_from_dmap(d, util.get_limbs())
  result = util.rebuild(d, k, util.get_connections(), 2, \
    util.get_grid(46), r.patch, 1)
  # for human in result:
  #   for i in range(14):
  #     print(i, human[i*3], human[i*3+1], human[i*3+2])
  print(result)
  final = util.format_annos(result, 'aaaaa')
  print(final)

  # new_k = util.get_key_map_from_dmap()
  # diff = np.max()

  # d = d[0]
  # util.vis_dmap(d, 'reader_dir_sample.jpg')

  # k = k[0]
  # k = np.max(k, axis=2)
  # misc.imsave('key.jpg', k)

