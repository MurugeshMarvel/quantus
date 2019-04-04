from data_aug.data_aug import *
from data_aug.bbox_util import *
import numpy as np 
import random
import cv2 
import matplotlib.pyplot as plt 
import pickle as pkl
import os
from tqdm import tqdm_notebook
import xml.etree.ElementTree as ET

DATA_DIR = '../data'
ANNOT_EXT = '.json'
IMG_EXT = '.jpg'
IMAGE_DIR = 'images'
ANNOT_DIR = 'annotations'

img_files = [img_file for img_file in [files for _, _, files in os.walk(os.path.join(DATA_DIR, IMAGE_DIR))][0] if img_file.endswith(IMG_EXT)]
annt_files = [annt_file for annt_file in [files for _, _, files in os.walk(os.path.join(DATA_DIR, ANNOT_DIR))][0] if annt_file.endswith(ANNOT_EXT)]

test_img = img_files[0]
test_annt = test_img.split('.')[0] + ANNOT_EXT

img_path = os.path.join(DATA_DIR, IMAGE_DIR, test_img)
annt_path = os.path.join(DATA_DIR, ANNOT_DIR, test_annt)

print (img_path)
print (annt_path)

img = cv2.imread(img_path)[:,:,::-1]
tree = ET.ElementTree(file="aug_img /img_100-6_aug_1.xml")
root = tree.getroot()