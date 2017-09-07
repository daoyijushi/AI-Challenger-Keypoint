import util
import pickle
import random
from scipy import misc
import numpy as np

class Reader:

  def __init__(self, img_dir, anno_path, batch_size, length=368):
    self.img_dir = img_dir
    with open(anno_path, 'rb') as f:
      self.data = pickle.load(f)
    self.batch_size = batch_size
    self.index = 0
    self.volumn = len(self.data)
    self.length = length
    random.shuffle(self.data)
    self.patch = util.normal_patch()
    self.limbs = util.limbs()
    print('Reader initialized. Data volumn %d, batch size %d.' \
      % (self.volumn, self.batch_size))

  def _resize(self, tmp, anno):
      h, w, _ = tmp.shape
      short_e = min(h, w)
      rate = self.length / short_e
      tmp = misc.imresize(tmp, rate)
      anno = np.round(np.array(anno) * rate)
      # crop and adjust annotations
      if short_e == h:
        # crop in w
        maxx = np.max(anno[:, ::3])
        minx = np.min(anno[:, ::3])
        mid = (maxx + minx) // 2
        left = int(mid - self.length // 2)
        right = int(mid + self.length // 2)
        if left < 0:
          overflow = -left
          right += overflow
          left = 0
        elif right > tmp.shape[1]:
          overflow = right - tmp.shape[1]
          left -= overflow
          right = tmp.shape[1]


        tmp = tmp[:, left:right, :]
        anno[:, ::3] -= left
      else:
        # crop in h
        maxy = np.max(anno[:, 1::3])
        miny = np.max(anno[:, 1::3])
        mid = (maxy + miny) // 2
        top = mid - self.length / 2
        bottom = mid + self.length / 2
        if top < 0:
          overflow = -top
          bottom += overflow
          top = 0
        elif bottom > tmp.shape[0]:
          overflow = bottom - tmp.shape[0]
          top -= overflow
          bottom = tmp.shape[0]
        tmp = tmp[top:bottom, :, :]
        anno[:, 1::3] -= top
      
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
      try:
        tmp = misc.imread(self.img_dir + piece['image_id'] + '.jpg')
        tmp, annos = self._resize(tmp, list(piece['keypoint_annotations'].values()))
        img.append(tmp)
        keypoint_hmap.append(util.get_key_hmap(tmp.shape, annos, self.patch))
        affinity_hmap.append(util.get_aff_hmap(tmp.shape, annos, self.limbs))
      except Exception as e:
        with open('reader.log', 'a') as f:
          f.write(piece['image_id'])
          f.write('\n')
          f.write(e)
          f.write('\n')
          
    img = np.array(img)
    keypoint_hmap = np.array(keypoint_hmap)
    affinity_hmap = np.array(affinity_hmap)
    return img, keypoint_hmap, affinity_hmap

# if __name__ == '__main__':
#   r = Reader('./data/train/', 'annotations.pkl', 128)
#   while True:
#     tic = tim
#     try:
#       img, kmap, amap = r.next_batch()

