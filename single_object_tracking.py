
import os

import torch
from torchvision import ops
import numpy as np
import cv2

def compute_iou(ground_truth, prediction_box):
  # Bounding box coordinates.
  ground_truth_bbox = torch.tensor([ground_truth], dtype=torch.float)
  prediction_bbox = torch.tensor([prediction_box], dtype=torch.float)
  
  # Get iou.
  iou = ops.box_iou(ground_truth_bbox, prediction_bbox)
  return iou.numpy()[0][0]

def success_rate (ground_truths, predictions, threshold = 0.5):
  total_frames = len(ground_truths)
  successful_frames = 0
  for i in range(total_frames):
    if ground_truths[i] == None or predictions[i] == None:
      successful_frames += (ground_truths == None and predictions[i] == None)
      continue
    iou = compute_iou(ground_truths[i], predictions[i])
    if (iou >= threshold):
       successful_frames += 1
  
  return successful_frames, successful_frames / total_frames


def tracking (data_dir, threshold = 0.5):

  image_dir = os.path.join(data_dir, 'img')
  image_files = sorted(os.listdir(image_dir))

  # Read the first image
  frame = cv2.imread(os.path.join(image_dir, image_files[0]))
  # Parse the ground truth annotation for the first frame
  gt_file = os.path.join(data_dir, 'groundtruth_rect.txt')
  with open(gt_file, 'r') as f:
    gt_annotations = [tuple(map(int, line.strip().split(','))) for line in f]
  (_, gt_x, gt_y, gt_w, gt_h, is_lost) = gt_annotations[0]

  # Initialize the tracker
  tracker = cv2.TrackerKCF_create()
  tracker.init(frame, (gt_x, gt_y, gt_w, gt_h))

  num_frames = len(image_files)
  iou_sum = 0
  ground_truths = []
  predictions = []

  # Loop through each image in the sequence
  for i in range(1, num_frames):
      # Read the image
      frame = cv2.imread(os.path.join(image_dir, image_files[i]))

      # Update the tracker
      success, bbox = tracker.update(frame)

      # Parse the ground truth annotation
      _, gt_x, gt_y, gt_w, gt_h, is_lost = gt_annotations[i]

      # Compute the intersection over union (IoU) between the predicted and ground truth bounding boxes
      if success:
          x, y, w, h = [int(k) for k in bbox]
          pred_box = [x, y, x + w, y + h]
          predictions.append(pred_box)
          if is_lost:
            ground_truths.append(None)
            continue
          gt_box = [gt_x, gt_y, gt_x + gt_w, gt_y + gt_h]
          
          ground_truths.append(gt_box)
      else:
        predictions.append(None)
        if not is_lost:
          gt_box = [gt_x, gt_y, gt_x + gt_w, gt_y + gt_h]
          ground_truths.append(gt_box)
        else:
          ground_truths.append(None)

      # Draw bounding box and ground truth annotation
      # if success:
      #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
      # cv2.rectangle(frame, (gt_x, gt_y), (gt_x + gt_w, gt_y + gt_h), (0, 0, 255), 2)

      # #Display the image
      # cv2.imshow('Frame', frame)

      # Exit if ESC pressed
      if cv2.waitKey(1) == 27:
          return -1, -1


  # Release resources
  cv2.destroyAllWindows()
  succ_frames, succ_rate = success_rate(ground_truths, predictions, threshold)
  return num_frames, succ_frames, succ_rate

def runTLPAtrr():

  attributes = {
    'bc': 'background clutter',
    'fm': 'fast motion',
    'iv': 'illumination variation',
    'occ': 'partial occlusions',
    'ov': 'out of view',
    'sv': 'large scale variation'
    }
  sum_succ = {
    'bc': 0,
    'fm': 0,
    'iv': 0,
    'occ': 0,
    'ov': 0,
    'sv': 0
  }
  sum_frames = {
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

    num, num_suc, rate = tracking(path)
    if num == -1:
        break
    sum_succ[attr] += num_suc
    sum_frames[attr] += num
    print(f'%s %.2f' % (data, rate * 100))

  for attr, succ in sum_succ.items():
     frames = sum_frames[attr]
     rate = succ / frames * 100
     print(f'%s %.2f' % (attributes[attr], rate))
  succ_rate = sum_succ / sum_frames
  print(f'Success rate on TinyTLP: %.2f' % (succ_rate * 100))

def runTinyTLP():
  dir_datasets = './TinyTLP'

  all_data = os.listdir(dir_datasets)
  sum_frames = 0
  sum_succ = 0
  for data in all_data:
    path = os.path.join(dir_datasets, data)
    num, num_suc, rate = tracking(path)
    if num == -1:
        break
    sum_succ += num_suc
    sum_frames += num
    print(f'%s %.2f' % (data, rate * 100))

  succ_rate = sum_succ / sum_frames
  print(f'Success rate on TinyTLP: %.2f' % (succ_rate * 100))

if __name__ == '__main__':
  #runTinyTLP()
  runTLPAtrr()
"""
  to create requirements.txt, run:
  py -m pipreqs.pipreqs .
"""