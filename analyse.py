from scipy import misc
import os
import matplotlib.pyplot as plt

src_dir = ('./data/')
names = os.listdir(src_dir)
print(len(names))

def summarize_size():
  x = []
  y = []
  for i, name in enumerate(names):
    img = misc.imread(src_dir + name)
    x.append(img.shape[0])
    y.append(img.shape[1])
    if (i + 1) % 1000 == 0:
      print("Process %d/210 ..." % ((i + 1) / 1000))

  print('Finish counting, plotting...')
  plt.scatter(x, y)
  plt.show()

