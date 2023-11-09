from __future__ import division

from models import *
from utils import *
from datasets import *

import os
import sys
import time
import datetime
import argparse
import cv2
from google.colab.patches import cv2_imshow
from skimage import io
from PIL import Image 
import matplotlib.pylab as plt


import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

font = cv2.FONT_HERSHEY_SIMPLEX

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--image_folder", type=str, default="KITTI_dataset/Imgs/", help="path to dataset")
	parser.add_argument("--model_def", type=str, default=os.path.join('config','yolov3_1class.cfg'), help="path to model definition file")
	parser.add_argument("--weights_path", type=str, default=os.path.join('weights', 'yolov3_COCO.weights'), help="path to weights file")
	parser.add_argument("--class_path", type=str, default=os.path.join('data', 'KITTI', 'classes.names'), help="path to class label file")
	parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
	parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
	parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
	parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
	parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
	opt = parser.parse_args()
	print(opt)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Set up model
	model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

	if opt.weights_path.endswith(".weights"):
		# Load darknet weights
		model.load_darknet_weights(opt.weights_path)
	else:
		# Load checkpoint weights
		model.load_state_dict(torch.load(opt.weights_path))

	model.eval()  # Set in evaluation mode

	classes = load_classes(opt.class_path)  # Extracts class labels from file

	Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

	dataloader = DataLoader(
		ImageFolder_KITTI(opt.image_folder, img_size=opt.img_size),
		batch_size=opt.batch_size,
		shuffle=False,
		num_workers=opt.n_cpu,)

	imgs = []  # Stores image paths
	img_detections = []  # Stores detections for each image index


	print("\nPerforming object detection:")
	img_cnt=1
	for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
		# Configure input
		input_imgs = Variable(input_imgs.type(Tensor))

		# Get detections
		with torch.no_grad():
			detections = model(input_imgs)
			detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

		
    
		for idx, img_detections in enumerate(detections):
			# Read image
			img = cv2.imread(str(img_paths[0]), cv2.IMREAD_COLOR)

			# Detected Bounding Boxes and Predicted Labels
			if img_detections is not None:
				# Rescale boxes to original image
				img_detections = rescale_boxes(img_detections, opt.img_size, img.shape[:2])
				unique_labels = img_detections[:, -1].cpu().unique()
				n_cls_preds = len(unique_labels)

				for x1, y1, x2, y2, conf, cls_conf, cls_pred in img_detections:
					box_w = x2 - x1
					box_h = y2 - y1

					# Draw Detected Bounding Boxes
					cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)

					bbox_name = classes[int(cls_pred)] + ': ' + str(int(cls_conf*100)) + "%"
					cv2.putText(img, bbox_name, (int(x1),int(y1)), font, 0.5, (255,0,0), 1, cv2.LINE_AA)
			# ----------------------------------------------------------------- #
			# ----- Save/Show the images with the generated bouding boxes ----- #
			# ----------------------------------------------------------------- #
				cv2.imwrite('/content/gdrive/MyDrive/YOLO/Results_weights/image_'+str(img_cnt)+'.png',img)
				img_cnt +=1
	print("Resultados guardados na pasta results")
	print("Numero de imagens onde foram detectados carros: %3d" % img_cnt)




