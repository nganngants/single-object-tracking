from tkinter import filedialog
import tkinter
import torch
from torchvision import ops
import numpy as np
import cv2
from tracker import tracking_demo


def demo():
    """
    - Allow users to choose video from their computer
    - Pass the video path to tracking demo
    """
    root = tkinter.Tk()
    root.withdraw()

    # Open file dialog to choose video file
    video_file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
    if video_file_path == '':
        print("No video file selected. Exiting...")
        exit()
    tracking_demo(video_file_path)
    


    
