import json

def load_data(file_name):
  with open(file_name) as file:
    data = json.load(file)
    print(data[0:10])
  return data

if __name__ == '__main__':
  load_data('annotations.json')
