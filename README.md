<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

# Single Object Tracking

This is our final project for CS231 - Introduction to Computer Vision course

## Overview

* This project experiments with the CSRT tracker from OpenCV on the TLP dataset (including both TinyTLP and TLPAttr) and compares its performance with other trackers.
* Additionally, the project allows for video demos on any selected video from the computer.
* Some demo results on the TLPAttr dataset can be viewed [here](https://youtu.be/YJgxtjXVfV4)

## Members

|**Student ID**|  **Full Name**  |       **Email**      |
|--------|-----------------|----------------------|
|21520069|Võ Trần Thu Ngân|21520069@gm.uit.edu.vn|
|21520474|  Lê Hoài Thương |21520474@gm.uit.edu.vn|

## Course Information

* University: University of Information Technology - VNUHCM UIT.
* Faculty: Computer Science
* Class: CS231.N21.KTHN
* Teacher: DSc. Mai Tiến Dũng

## Code

### Prerequisits

1. Clone the repo
    ```
   git clone https://github.com/nganngants/single-object-tracking.git
    ```
2. Download the dataset [here](https://amoudgl.github.io/tlp/) and change the datasets path in `single_object_tracking.py`
3. Install the python dependency packages.
    ```
    pip install -r requirements.txt
    ```

### Usage
Terminal command:
  ```
  py single_object_tracking.py -eval [data] -show [flag]
  ```
  
  * To evaluate the tracker on TLP, TinyTLP or TLPAttr dataset, replace the `[data]` with `tlp`, `tiny-tlp` or `tlp-attr` respectively. Replace `[flag]` with `True` if you want to display the videos during evaluation process; otherwise, use `False`.
  * To demo on any video, replace the `[data]` with `demo`. No need to specify the `-show` argument.

# Acknowledgment

I would like to thank the creators of the CSRT tracker from OpenCV for their valuable tool and the creators of the TLP dataset for providing the necessary data for comprehensive evaluations and comparisons.
