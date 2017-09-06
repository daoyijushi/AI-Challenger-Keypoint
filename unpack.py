import os

for i in range(210):
  line = 'rm -rf ./%d' % i
  os.system(line)

