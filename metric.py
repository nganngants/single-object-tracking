import torch
from torchvision import ops
import numpy as np

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