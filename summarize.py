from scipy import misc
import os
import matplotlib.pyplot as plt
import json
import pickle
import numpy as np


src_dir = './data/'
save_dir = './summary/'

def sum_img_size():
  names = os.listdir(src_dir)
  print(len(names))
  x = []
  y = []
  for i, name in enumerate(names):
    img = misc.imread(src_dir + name)
    x.append(img.shape[0])
    y.append(img.shape[1])
    if (i + 1) % 1000 == 0:
      print("Process %d/210 ..." % ((i + 1) / 1000))

  print('Finish counting, plotting...')
  plt.scatter(x, y, s=1)
  plt.show()

def sum_img_color():
  names = os.listdir(src_dir)
  num = len(names)
  print(num)
  color = np.zeros(3)
  color_s = np.zeros(3)
  error = []
  for i, name in enumerate(names):
    try:
      img = misc.imread(src_dir + name)
      avg = np.mean(img, axis=(0, 1))
      color += avg
      color_s += np.square(avg)
      if (i + 1) % 1000 == 0:
        print("Process %d/210 ..." % ((i + 1) / 1000))
    except:
      print('Got error at %s' % name)
      error.append(name)
  mean = color / num
  mean_s = color_s / num
  var = mean_s - np.square(mean)
  print("Mean:", mean)
  print("Var:", var)
  np.save('mean.npy', mean)
  np.save('var.npy', var)
  if len(error) != 0:
    with open('error.log', 'w') as f:
      for name in error:
        f.write(name)
        f.write('\n')

def sum_p_cnt(data):
  # how many people in one image (at most 11)
  p_cnt = [0] * 12
  for piece in data:
    cnt = len(piece['human_annotations'])
    p_cnt[cnt] += 1

  print('Saving people count...')
  with open(save_dir + 'person_cnt.txt', 'w') as f:
    for num in p_cnt:
      f.write(str(num))
      f.write(',')

def sum_p_pos(data):
  # where are the people
  p_pos_x = []
  p_pos_y = []
  for piece in data:
    for pos in piece['human_annotations'].values():
      p_pos_x.append((pos[0] + pos[2]) / 2)
      p_pos_y.append((pos[1] + pos[3]) / 2)
  print('Showing people position...')
  plt.title("People position")
  plt.scatter(p_pos_x, p_pos_y, s=1)
  plt.show()

def sum_p_size(data):
  # how big are the people
  p_sz_x = []
  p_sz_y = []
  for piece in data:
    for pos in piece['human_annotations'].values():
      p_sz_x.append(pos[2] - pos[0])
      p_sz_y.append(pos[3] - pos[1])
  print('Showing people size...')
  plt.title("People size")
  plt.scatter(p_sz_x, p_sz_y, s=1)
  plt.show()

def sum_k_pos(data):
  # where are the key points
  k_pos_x = []
  k_pos_y = []
  for i in range(14):
    k_pos_x.append([])
    k_pos_y.append([])
  for piece in data:
    for pos in piece['keypoint_annotations'].values():
      for i in range(14):
        k_pos_x[i].append(pos[i * 3])
        k_pos_y[i].append(pos[i * 3 + 1])
  print('Showing keypoint position...')
  plt.title("Keypoint position")
  plt.scatter(k_pos_x, k_pos_y, s=1)
  plt.show()

def sum_k_cnt(data):
  # how many key points are expected to be seen (14 different pts in total)
  k_cnt = [[0] * 14, [0] * 14, [0] * 14]
  for piece in data:
    for pos in piece['keypoint_annotations'].values():
      for i in range(14):
        k_cnt[pos[i * 3 + 2] - 1][i] += 1
  print('Saving keypoint count...')
  with open(save_dir + 'keypoint_cnt.txt', 'w') as f:
    for k in k_cnt:
      for num in k:
        f.write(str(num))
        f.write(',')
      f.write('\n')

def sum_anno(file_name):
  with open(file_name, 'r') as f:
    data = json.load(f)

  pickle_name = file_name.split('.')[0] + '.pkl'
  with open(pickle_name, 'wb') as f:
    pickle.dump(data, f)

if __name__ == '__main__':
  sum_img_color()
