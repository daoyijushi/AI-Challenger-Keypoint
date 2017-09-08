import datetime
import reader
import pickle
import sys
import util

repeat = 210
r = reader.Reader('./data/train/', 'annotations.pkl', 1000)
record = []
cnt = 0
while True:
  tic = datetime.datetime.now()
  img, kmap, amap = r.next_batch()
  toc = datetime.datetime.now()
  diff = (toc - tic).microseconds
  record.append(diff)
  print('Process %d/210, time cost %g' % (cnt, diff * 1e-6))
  cnt += 1
  if cnt == repeat:
    break
  img = img[0]
  kmap = kmap[0]
  amap = amap[0]
  util.visualization(img, kmap, amap, 'sample%d.jpg'%cnt)

with open('intervals.pkl', 'wb') as f:
  pickle.dump(diff, f)

avg = sum(record) / len(record) / 1000
print(avg)
