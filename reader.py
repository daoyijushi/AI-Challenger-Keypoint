import util
import pickle
import random
from scipy import misc

class reader:

  def __init__(img_dir, annotations, batch_size):
    self.img_dir = img_dir
    with open(annotations, 'rb') as f:
      self.data = pickle.load(f)
    self.batch_size = batch_size
    self.index = 0
    self.volumn = len(data)
    self.data = random.shuffle(self.data)
    self.patch = util.normal_patch()
    print('Reader initialized. Data volumn %d, batch size %d.' \
      % (self.volumn, self.batch_size))

  def next_batch():
    print("Sampling batch from %d..." % self.index)
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
      tmp = misc.imread()
      keypoint_hmap.append(util.get_key_hmap())
      # rodo: resize, affinity


