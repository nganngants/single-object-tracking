import os

import cv2
from metric import success_rate

def tracking (data_dir, show = False, threshold = 0.5):

  image_dir = os.path.join(data_dir, 'img')
  image_files = sorted(os.listdir(image_dir))
  if len(image_files) == 0:
     print(f'%s does not have any image')

  frame = cv2.imread(os.path.join(image_dir, image_files[0]))
  # Parse the ground truth annotation for the first frame
  gt_file = os.path.join(data_dir, 'groundtruth_rect.txt')
  with open(gt_file, 'r') as f:
    gt_annotations = [tuple(map(int, line.strip().split(','))) for line in f]
  (_, gt_x, gt_y, gt_w, gt_h, is_lost) = gt_annotations[0]

  tracker = cv2.TrackerCSRT_create()
  tracker.init(frame, (gt_x, gt_y, gt_w, gt_h))

  num_frames = len(image_files)
  ground_truths = []
  predictions = []

  for i in range(1, num_frames):
      frame = cv2.imread(os.path.join(image_dir, image_files[i]))
      _, gt_x, gt_y, gt_w, gt_h, gt_is_lost = gt_annotations[i]

      success, bbox = tracker.update(frame)
      if success:
          x, y, w, h = [int(k) for k in bbox]
          pred_box = [x, y, x + w, y + h]
          predictions.append(pred_box)
          if gt_is_lost == 1:
            ground_truths.append(None)
          else:
            gt_box = [gt_x, gt_y, gt_x + gt_w, gt_y + gt_h]
            
            ground_truths.append(gt_box)
      else:
        predictions.append(None)
        if gt_is_lost == 0:
          gt_box = [gt_x, gt_y, gt_x + gt_w, gt_y + gt_h]
          ground_truths.append(gt_box)
        else:
          ground_truths.append(None)
      if show:
        if success and x >= 0 and y >= 0:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if gt_is_lost == 0:
           cv2.rectangle(frame, (gt_x, gt_y), (gt_x + gt_w, gt_y + gt_h), (0, 0, 255), 2)

        cv2.imshow('Frame', frame)

      # Exit if ESC pressed
      if cv2.waitKey(1) == 27:
          return -1, -1, -1


  # Release resources
  cv2.destroyAllWindows()
  succ_frames, succ_rate = success_rate(ground_truths, predictions, threshold)
  return num_frames, succ_frames, succ_rate

def tracking_demo(video_file_path):

    cap = cv2.VideoCapture(video_file_path)
    cv2.namedWindow("Tracking")
    success, img = cap.read()
    bbox = cv2.selectROI("Tracking", img, False)
    tracker = cv2.TrackerCSRT_create()
    tracker.init(img, bbox)

    while True:
       success, frame = cap.read()
       if success == False:
          break
       success, bbox = tracker.update(frame)
       if success:
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
       cv2.imshow("Tracking", frame)
       if cv2.waitKey(1) == 27:
          break
    cv2.destroyAllWindows()
