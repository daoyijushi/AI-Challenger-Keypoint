import util
import pickle
import random
from scipy import misc
import numpy as np
import random

class Reader:

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
    self.patch = util.get_patch(pl,ps)
    self.limbs = util.get_limbs()
    self.mean = np.array([122.35131039, 115.17054545, 107.60200075])
    self.var = np.array([35.77071304, 35.39201422, 37.7260754])
    np.random.seed(822)
    print('Reader initialized. Data volumn %d, batch size %d.' \
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
    affinity_hmap = []
    names = []
    for piece in data_batch:
      tmp = misc.imread(self.img_dir + piece['image_id'] + '.jpg')
      names.append(piece['image_id'])

      tmp, rate, left, top = \
        util.resize(tmp, self.length)
      annos = np.array(list(piece['keypoint_annotations'].values()), dtype=np.float16)
      annos[:, ::3] = annos[:, ::3] * rate - left
      annos[:, 1::3] = annos[:, 1::3] * rate - top
      annos = annos.astype(np.int16)
      img.append(tmp)

      tmp = misc.imresize(tmp, (self.short, self.short))
      rate = self.short / self.length
      annos[:, ::3] = annos[:, ::3] * rate
      annos[:, 1::3] = annos[:, 1::3] * rate
      annos = np.round(annos).astype(np.int8)


      keypoint_hmap.append(util.get_key_hmap(tmp.shape, annos, self.patch, self.patch_l//2))
      affinity_hmap.append(util.get_aff_hmap(tmp.shape, annos, self.limbs))

    img = np.array(img, dtype=np.float32)
    img -= self.mean
    img /= self.var
    keypoint_hmap = np.array(keypoint_hmap, dtype=np.float32)
    affinity_hmap = np.array(affinity_hmap, dtype=np.float32)
    return img, keypoint_hmap, affinity_hmap, names

class StrictReader:

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
    self.patch = util.get_patch(10,4)
    self.limbs = util.get_limbs()
    self.mean = np.array([122.35131039, 115.17054545, 107.60200075])
    self.var = np.array([35.77071304, 35.39201422, 37.7260754])
    np.random.seed(822)
    print('Reader initialized. Data volumn %d, batch size %d.' \
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
    affinity_hmap = []
    names = []
    for piece in data_batch:
      tmp = misc.imread(self.img_dir + piece['image_id'] + '.jpg')
      names.append(piece['image_id'])

      tmp, rate, left, top = \
        util.resize(tmp, self.length)
      annos = np.array(list(piece['keypoint_annotations'].values()), dtype=np.float16)
      annos[:, ::3] = annos[:, ::3] * rate - left
      annos[:, 1::3] = annos[:, 1::3] * rate - top
      annos = annos.astype(np.int16)
      img.append(tmp)

      tmp = misc.imresize(tmp, (self.short, self.short))
      rate = self.short / self.length
      annos[:, ::3] = annos[:, ::3] * rate
      annos[:, 1::3] = annos[:, 1::3] * rate
      annos = np.round(annos).astype(np.int8)


      keypoint_hmap.append(util.get_key_hmap(tmp.shape, annos, self.patch, self.patch_l//2, strict=True))
      affinity_hmap.append(util.get_aff_hmap(tmp.shape, annos, self.limbs, strict=True))

    img = np.array(img, dtype=np.float32)
    img -= self.mean
    img /= self.var
    keypoint_hmap = np.array(keypoint_hmap, dtype=np.float32)
    affinity_hmap = np.array(affinity_hmap, dtype=np.float32)
    return img, keypoint_hmap, affinity_hmap, names

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
    names = []
    for piece in data_batch:
      tmp = misc.imread(self.img_dir + piece['image_id'] + '.jpg')
      names.append(piece['image_id'])
      
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
    return img, keypoint_hmap, direction_hmap, names

class KReader:

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
    self.mean = np.array([122.35131039, 115.17054545, 107.60200075])
    self.var = np.array([35.77071304, 35.39201422, 37.7260754])
    np.random.seed(822)
    print('KReader initialized. Data volumn %d, batch size %d.' \
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
    names = []
    for piece in data_batch:
      tmp = misc.imread(self.img_dir + piece['image_id'] + '.jpg')
      names.append(piece['image_id'])
      
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

    img = np.array(img, dtype=np.float32)
    img -= self.mean
    img /= self.var
    keypoint_hmap = np.array(keypoint_hmap, dtype=np.float32)
    return img, keypoint_hmap, names

class LReader:
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
    self.limb_index = [0,1,2,3,4,5,6,7,8,9,10,11,12]
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
    names = []
    lables = []
    link_kmaps = []
    for piece in data_batch:
      tmp = misc.imread(self.img_dir + piece['image_id'] + '.jpg')
      names.append(piece['image_id'])
      
      # resize the img and annotation to 368x368
      tmp, rate, left, top = \
        util.resize(tmp, self.length)
      annos = np.array(list(piece['keypoint_annotations'].values()), dtype=np.float16)
      annos[:, ::3] = annos[:, ::3] * rate - left
      annos[:, 1::3] = annos[:, 1::3] * rate - top
      annos = annos.astype(np.int16)

      img.append(tmp)

      # resize the img and annotations to 46x46
      tmp = misc.imresize(tmp, (self.short, self.short))
      rate = self.short / self.length
      annos[:, ::3] = annos[:, ::3] * rate
      annos[:, 1::3] = annos[:, 1::3] * rate
      annos = np.round(annos).astype(np.int8)

      kmap = util.get_key_hmap(\
        tmp.shape, annos, self.patch, self.patch_l//2)
      keypoint_hmap.append(kmap)

      human_num = len(annos)
      failed = True
      # since many img contains only one people, we manually
      # reduce the possibility to get the positive sample

      random.shuffle(self.limb_index)
      if np.random.random_sample() < 0.15 or human_num == 1:
        # extract from the same people
        for limb in self.limb_index:
          start = self.limbs[limb][0]
          end = self.limbs[limb][1]
          human = annos[np.random.randint(human_num)]
          start_x = human[start*3]
          start_y = human[start*3+1]
          start_v = human[start*3+2]
          end_x = human[end*3]
          end_y = human[end*3+1]
          end_v = human[end*3+2]
          if util.validate(start_x, start_y, start_v, self.short, self.short) \
            and util.validate(end_x, end_y, end_v, self.short, self.short):

            kmap_start = util.get_single_kmap((self.short, self.short), start_x, start_y, self.patch_l//2, self.patch)
            kmap_end = util.get_single_kmap((self.short, self.short), end_x, end_y, self.patch_l//2, self.patch)
            link_kmaps.append(np.stack((kmap_start, kmap_end), axis=-1))
            failed = False
            break
        if failed:
          link_kmaps.append(np.zeros((self.short, self.short, 2)))
          print('no validate limb in %s (same)' % piece['image_id'])

        lables.append(1)

      else:
        # extract from two different humen
        for limb in self.limb_index:
          start = self.limbs[limb][0]
          end = self.limbs[limb][1]

          index1 = np.random.randint(human_num)
          human1 = annos[index1]
          start_x = human1[start*3]
          start_y = human1[start*3+1]
          start_v = human1[start*3+2]

          index2 = np.random.randint(human_num)
          human2 = annos[index2]
          end_x = human2[end*3]
          end_y = human2[end*3+1]
          end_v = human2[end*3+2]

          if util.validate(start_x, start_y, start_v, self.short, self.short) \
            and util.validate(end_x, end_y, end_v, self.short, self.short) \
            and index1 != index2:
            
            kmap_start = util.get_single_kmap((self.short, self.short), start_x, start_y, self.patch_l//2, self.patch)
            kmap_end = util.get_single_kmap((self.short, self.short), end_x, end_y, self.patch_l//2, self.patch)
            link_kmaps.append(np.stack((kmap_start, kmap_end), axis=-1))
            lables.append(0)
            failed = False
            break
        if failed:
          link_kmaps.append(np.zeros((self.short, self.short, 2)))
          print('no validate limb in %s (differen)' % piece['image_id'])

        lables.append(0)

      tmp_dmap, tmp_dmap_re = util.get_dir_hmap(\
        tmp.shape, annos, self.ones, self.limbs, self.patch_l//2)

      dmap = util.weight_dir_hmap(kmap, tmp_dmap, tmp_dmap_re, self.limbs)

      direction_hmap.append(dmap)

    img = np.array(img, dtype=np.float32)
    img -= self.mean
    img /= self.var
    keypoint_hmap = np.array(keypoint_hmap, dtype=np.float32)
    direction_hmap = np.array(direction_hmap, dtype=np.float32)
    lables = np.array(lables, dtype=np.float32)
    link_kmaps = np.array(link_kmaps, dtype=np.float32)
    
    return img, keypoint_hmap, direction_hmap, names, lables, link_kmaps

class DAReader:
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
    self.ones = np.ones((self.patch_l,self.patch_l,2))
    self.patch = util.get_patch(10,4)
    self.limbs = util.get_limbs()
    self.mean = np.array([122.35131039, 115.17054545, 107.60200075])
    self.var = np.array([35.77071304, 35.39201422, 37.7260754])
    np.random.seed(822)
    print('Reader initialized. Data volumn %d, batch size %d.' \
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
    direction_hmap = []
    affinity_hmap = []
    names = []
    for piece in data_batch:
      tmp = misc.imread(self.img_dir + piece['image_id'] + '.jpg')
      names.append(piece['image_id'])

      tmp, rate, left, top = \
        util.resize(tmp, self.length)
      annos = np.array(list(piece['keypoint_annotations'].values()), dtype=np.float16)
      annos[:, ::3] = annos[:, ::3] * rate - left
      annos[:, 1::3] = annos[:, 1::3] * rate - top
      annos = annos.astype(np.int16)
      img.append(tmp)

      tmp = misc.imresize(tmp, (self.short, self.short))
      rate = self.short / self.length
      annos[:, ::3] = annos[:, ::3] * rate
      annos[:, 1::3] = annos[:, 1::3] * rate
      annos = np.round(annos).astype(np.int8)

      kmap = util.get_key_hmap(tmp.shape, annos, self.patch, self.patch_l//2)
      tmp_dmap, tmp_dmap_re = util.get_dir_hmap(\
        tmp.shape, annos, self.ones, self.limbs, self.patch_l//2)
      dmap = util.weight_dir_hmap(kmap, tmp_dmap, tmp_dmap_re, self.limbs)
      direction_hmap.append(dmap)

      affinity_hmap.append(util.get_aff_hmap(tmp.shape, annos, self.limbs))

    img = np.array(img, dtype=np.float32)
    img -= self.mean
    img /= self.var
    direction_hmap = np.array(direction_hmap, dtype=np.float32)
    affinity_hmap = np.array(affinity_hmap, dtype=np.float32)
    return img, direction_hmap, affinity_hmap, names

class MultiReader:
  def __init__(self, img_dir, anno_path, batch_size, pl=10, ps=4):
    self.img_dir = img_dir

    with open(anno_path, 'rb') as f:
      self.data = pickle.load(f)
    random.shuffle(self.data)

    self.batch_size = batch_size
    self.index = 0
    self.volumn = len(self.data)

    self.micro = 46
    self.tiny = 92
    self.mid = 184
    self.large = 368

    self.patch_l = pl
    self.patch_s = ps
    self.patch = util.get_patch(self.patch_l, self.patch_s)

    self.ones = np.ones((self.patch_l,self.patch_l,2))
    self.limbs = util.get_limbs()
    self.mean = np.array([122.35131039, 115.17054545, 107.60200075])
    self.var = np.array([35.77071304, 35.39201422, 37.7260754])

    np.random.seed(822)
    
    print('DirReader initialized. Data volumn %d, batch size %d.' \
      % (self.volumn, self.batch_size))

  def append_dmap(self, annos, container, length):
    k = util.get_key_hmap((length, length), annos, self.patch, self.patch_l//2)
    tmp, tmp_re = util.get_dir_hmap((length, length), annos, self.ones, self.limbs, self.patch_l//2)
    d = util.weight_dir_hmap(k, tmp, tmp_re, self.limbs)
    container.append(d)

  def next_batch(self):
    start = self.index
    end = self.index + self.batch_size
    if end >= self.volumn:
      end = self.volumn
      self.index = 0
    else:
      self.index = end

    data_batch = self.data[start:end]
    imgs = []
    dmaps_micro = []
    dmaps_tiny = []
    dmaps_mid = []
    dmaps_large = []
    names = []

    for piece in data_batch:
      tmp = misc.imread(self.img_dir + piece['image_id'] + '.jpg')
      names.append(piece['image_id'])
      
      # resize to 368x368
      tmp, rate, left, top = \
        util.resize(tmp, self.large)
      imgs.append(tmp)


      annos = np.array(list(piece['keypoint_annotations'].values()), dtype=np.float16)

      annos[:, ::3] = annos[:, ::3] * rate - left
      annos[:, 1::3] = annos[:, 1::3] * rate - top
      annos = annos.astype(np.int16)
      self.append_dmap(annos, dmaps_large, self.large)

      annos = annos.astype(np.float32)
      annos[:, ::3] = annos[:, ::3] * 0.5
      annos[:, 1::3] = annos[:, 1::3] * 0.5
      annos = np.round(annos).astype(np.int16)
      self.append_dmap(annos, dmaps_mid, self.mid)

      annos = annos.astype(np.float32)
      annos[:, ::3] = annos[:, ::3] * 0.5
      annos[:, 1::3] = annos[:, 1::3] * 0.5
      annos = np.round(annos).astype(np.int16)
      self.append_dmap(annos, dmaps_tiny, self.tiny)

      annos = annos.astype(np.float32)
      annos[:, ::3] = annos[:, ::3] * 0.5
      annos[:, 1::3] = annos[:, 1::3] * 0.5
      annos = np.round(annos).astype(np.int16)
      self.append_dmap(annos, dmaps_micro, self.micro)


    imgs = np.array(imgs, dtype=np.float32)
    imgs -= self.mean
    imgs /= self.var
    dmaps_micro = np.array(dmaps_micro)
    dmaps_tiny = np.array(dmaps_tiny)
    dmaps_mid = np.array(dmaps_mid)
    dmaps_large = np.array(dmaps_large)
    return imgs, dmaps_micro, dmaps_tiny, dmaps_mid, dmaps_large, names

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
  i, k, d, names = r.next_batch()

  k = k[0]
  d = d[0]
  i = i[0]

  d = np.minimum(d, 0)
  s = np.sum(d)
  print(s)

  # k = util.get_kmap_from_dmap(d, util.get_limbs())
  # result = util.rebuild(d, k, util.get_connections(), 2, \
  #   util.get_grid(46), r.patch, 1)
  # # for human in result:
  # #   for i in range(14):
  # #     print(i, human[i*3], human[i*3+1], human[i*3+2])
  # print(result)
  # final = util.format_annos(result, 'aaaaa')
  # print(final)

  # new_k = util.get_key_map_from_dmap()
  # diff = np.max()

  # d = d[0]
  # util.vis_dmap(d, 'reader_dir_sample.jpg')

  # k = k[0]
  # k = np.max(k, axis=2)
  # misc.imsave('key.jpg', k)

