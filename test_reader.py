import datetime
import reader
import pickle

r = Reader('./data/train/', 'annotations.pkl', 128)
record = []
while True:
  tic = datetime.datetime.now()
  img, kmap, amap = r.next_batch()
  toc = datetime.datetime.now()
  diff = (toc - tic).microseconds
  record.append(diff)

with open('intervals.pkl', 'wb') as f:
  pickle.dump(diff, f)

avg = sum(diff) / len(diff)
print(avg)
