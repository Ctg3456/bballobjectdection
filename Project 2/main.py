from ultralytics import YOLO
import git
import os
import torch
import utils
import subprocess

# cloning the yolov5 repo

git.Git("/Users/trevorcochran/Documents/Project 2").clone("https://github.com/ultralytics/yolov5")

os.chdir("/Users/trevorcochran/Documents/Project 2/yolov5")

print("The Current working directory is: {0}".format(os.getcwd()))

# training my custom dataset

os.system("python train.py --img 640 --batch 16 --epochs 75 --data bball_data.yaml --weights yolov5x.pt --nosave --cache")

# detection command on trained dataset output with the NBA game input file game5game.mp4

os.system("python detect.py --weights runs/train/exp2/weights/last.pt --img 640 --conf 0.25 --source ../game5game.mp4")



