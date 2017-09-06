import numpy as np
import os
from scipy import misc

data_dir = './data/'
names = os.listdir(data_dir)
errors = []

for i, name in enumerate(names):
  try:
    img = misc.imread(data_dir + name)
    n_name = name.split('.')[0] + '.npy'
    np.save(data_dir + n_name, img)
  except: # some of the uploaded images may be broken
    print('We got error at %s' % name)
    errors.append(name)
  if (i + 1) % 1000 == 0:
    print('Processing %d/210...' % ((i + 1) / 1000))

if len(errors) != 0:
  with open('img2npy_error.log') as f:
    for name in errors:
      f.write(name)
      f.write('\n')
