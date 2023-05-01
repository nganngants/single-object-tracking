
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


def tracking (data_dir):

  # List all the images in the directory
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
  tracker = cv2.legacy.TrackerKCF_create()
  tracker.init(frame, (gt_x, gt_y, gt_w, gt_h))

  # Initialize variables for tracking evaluation
  num_frames = len(image_files)
  iou_sum = 0

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
          if is_lost:
            continue
          x, y, w, h = [int(k) for k in bbox]
          pred_box = [x, y, x + w, y + h]
          gt_box = [gt_x, gt_y, gt_x + gt_w, gt_y + gt_h]
          iou = compute_iou(pred_box, gt_box)
          iou_sum += iou
      else:
          iou_sum += is_lost

      # Draw bounding box and ground truth annotation
      if success:
          cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
      cv2.rectangle(frame, (gt_x, gt_y), (gt_x + gt_w, gt_y + gt_h), (0, 0, 255), 2)

      # Display the image
      cv2.imshow('Frame', frame)

      # Exit if ESC pressed
      if cv2.waitKey(1) == 27:
          return -1, -1

  # Compute the tracking evaluation metrics
  avg_iou = iou_sum / (num_frames - 1)

  # Release resources
  cv2.destroyAllWindows()

  return num_frames, avg_iou

dir_datasets = './TinyTLP'
all_data = os.listdir(dir_datasets)
sum_frames = 0
sum_iou = 0.0
for data in all_data:
   path = os.path.join(dir_datasets, data)
   num, iou = tracking(path)
   if iou == -1:
      break
   sum_iou += iou * num
   sum_frames += num
   print(data, iou)

print(sum_iou / sum_frames)
"""
  to create requirements.txt, run:
  py -m pipreqs.pipreqs .
"""