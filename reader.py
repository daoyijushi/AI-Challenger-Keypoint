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
    # random.shuffle(self.data)
    self.patch = util.normal_patch()
    self.limbs = util.limbs()
    self.mean = np.array([122.35131039, 115.17054545, 107.60200075])
    self.var = np.array([35.77071304, 35.39201422, 37.7260754])
    np.random.seed(822)
    print('Reader initialized. Data volumn %d, batch size %d.' \
      % (self.volumn, self.batch_size))

  def _resize(self, tmp, anno):
      h, w, _ = tmp.shape
      anno = np.array(anno, dtype=np.float64)
      # print('init shape:', tmp.shape)
      # print('init anno', anno)
      if h < w:
        rate = self.length / h
        tmp = misc.imresize(tmp, (self.length, int(rate*w)))
        anno[:, ::3] *= rate
        anno[:, 1::3] *= rate
        # print('rescaled shape:', tmp.shape)
        # print('rescaled anno:', anno)
        # random crop in w
        if tmp.shape[1] > self.length:
          left = np.random.randint(0, tmp.shape[1] - self.length)
          right = left + self.length
          tmp = tmp[:, left:right, :]
          anno[:, ::3] -= left
          # print('crop from %d to %d' % (left, right))
          # print('croped anno', anno)
        else:
          # if this condition happens
          # tmp.shape must be only a little bit different
          # from expected, so a simple resize is enough
          tmp = misc.imresize(tmp, (self.length, self.length))
      else:
        rate = self.length / w
        tmp = misc.imresize(tmp, (int(rate*h), self.length))
        anno[:, ::3] *= rate
        anno[:, 1::3] *= rate
        # print('rescaled shape:', tmp.shape)
        if tmp.shape[0] > self.length:
          # print('rescaled anno:', anno)
          # random crop in h
          top = np.random.randint(0, tmp.shape[0] - self.length)
          bottom = top + self.length
          tmp = tmp[top:bottom, :, :]
          anno[:, 1::3] -= top
          # print('crop from %d to %d' % (top, bottom))
          # print('croped anno', anno)
        else:
          tmp = misc.imresize(tmp, (self.length, self.length))
      
      # print('fnial shape:', tmp.shape)
      # print('final anno:', anno)
      return tmp, anno.astype(np.int16)

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
      # try:
      tmp = misc.imread(self.img_dir + piece['image_id'] + '.jpg')
      tmp, annos = self._resize(tmp, list(piece['keypoint_annotations'].values()))
      img.append(tmp)
      tmp = misc.imresize(tmp, (self.short, self.short))
      keypoint_hmap.append(util.get_key_hmap(tmp.shape, annos, self.patch))
      affinity_hmap.append(util.get_aff_hmap(tmp.shape, annos, self.limbs))
      # except Exception as e:
      #   print(e)
      #   with open('reader.log', 'a') as f:
      #     f.write(piece['image_id'])
      #     f.write('\n')
      #     f.write(str(e))
      #     f.write('\n')
          
    img = np.array(img, dtype=np.float64)
    img -= self.mean
    img /= self.var
    keypoint_hmap = np.array(keypoint_hmap)
    affinity_hmap = np.array(affinity_hmap)
    return img, keypoint_hmap, affinity_hmap

if __name__ == '__main__':
  r = Reader('./image/', 'anno_sample.pkl', 1)
  i, k, a = r.next_batch()
  i = i[0]
  k = k[0]
  a = a[0]
  util.visualization(i, k, a)

