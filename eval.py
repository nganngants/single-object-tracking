import os
import cv2
from tracker import tracking

def evalTLPAtrr(show=False):

  attributes = {
    'bc': 'background clutter',
    'fm': 'fast motion',
    'iv': 'illumination variation',
    'occ': 'partial occlusions',
    'ov': 'out of view',
    'sv': 'large scale variation'
    }
  cnt_succ = {
    'bc': 0,
    'fm': 0,
    'iv': 0,
    'occ': 0,
    'ov': 0,
    'sv': 0
  }
  cnt_frames = {
    'bc': 0,
    'fm': 0,
    'iv': 0,
    'occ': 0,
    'ov': 0,
    'sv': 0
  }
  dir_datasets = './TLPAttr'
  all_data = os.listdir(dir_datasets)
  for data in all_data:
    path = os.path.join(dir_datasets, data)
    attr = data.split('_')[0]
    num, num_succ, rate = tracking(path, show)
    if num == -1:
        exit()
    cnt_succ[attr] += num_succ
    cnt_frames[attr] += num
    print(f'%s %.2f' % (data, rate * 100))

  sum_succ = 0
  sum_frames = 0
  for attr, succ in cnt_succ.items():
     frames = cnt_frames[attr]
     if frames == 0:
        continue
     rate = succ / frames * 100
     print(f'%s %.2f' % (attributes[attr], rate))
     sum_succ += succ
     sum_frames += frames
  succ_rate = sum_succ / sum_frames
  print(f'Success rate on TLPAtrr: %.2f' % (succ_rate * 100))

def evalTinyTLP(show=False):
  dir_datasets = './TinyTLP'
  all_data = os.listdir(dir_datasets)
  sum_frames = 0
  sum_succ = 0
  for data in all_data:
    path = os.path.join(dir_datasets, data)
    num, num_suc, rate = tracking(path, show)
    if num == -1:
        exit()
    sum_succ += num_suc
    sum_frames += num
    print(f'%s %.2f' % (data, rate * 100))

  succ_rate = sum_succ / sum_frames
  print(f'Success rate on TinyTLP: %.2f' % (succ_rate * 100))