# Python's json module cannot parse correctly
# when "," follows the last element,
# so we simply remove all the "," before "}"

import sys

def compress(content):
  content = content.replace(' ', '')
  content = content.replace('\n', '')
  content = content.replace('\t', '')
  return content

def reformat(src, target):
  content = None
  with open(src, 'r') as file:
    content = compress(file.read())

  start = 0
  while True:
    pos = content.find('}', start)
    if pos == -1:
      break
    else:
      if content[pos - 1] == ',':
        content = content[:pos - 1] + content[pos:]
        start = pos
      else:
        start = pos + 1

  with open(target, 'w') as file:
    file.write(content)

if __name__ == '__main__':
  reformat('sample.json', 'data.json')