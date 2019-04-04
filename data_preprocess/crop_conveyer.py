from skimage import draw
import numpy as np
import cv2
import json
import warnings
import os
import sys


def poly2mask(img,blobs, mask_img_dir, label, idx, name):
    HEIGHT, WIDTH = img.shape[:2]
    mask = np.zeros((HEIGHT, WIDTH), dtype= np.uint8)
    for l in blobs:
        fill_row_coords, fill_col_coords = draw.polygon(l[1], l[0], l[2])
        mask[fill_row_coords, fill_col_coords] = 1
    img[:,:,0] *=  mask
    img[:,:,1] *=  mask
    img[:,:,2] *=  mask
    cv2.imwrite(mask_img_dir + "/"  + name + ".jpg", img)
    print ('saved')

def json_to_mask(json_dir, mask_img_dir, img):
    HEIGHT, WIDTH = img.shape[:2]
    name = json_dir.split('/')[-1].split('.')[0]
    obj = json.load(open(json_dir,'r'))
    classes = {}
    annotations = obj['shapes']
    print (name)
    blob = {}

    for annot in annotations:
        blobs = []
        label = annot['label']
        if (label != ''):
            if label not in classes:
                classes[label] = 0
                classes['labelname'] = label
                blob[label] = []
            points = annot['points']
            x_coord = []
            y_coord = []
            l = []
            for p in points:
                x_coord.append(p[0])
                y_coord.append(p[1])
            shape = (HEIGHT, WIDTH)
            l.append(x_coord)
            l.append(y_coord)
            l.append(shape)
            blob[label].append(l)
        for label in blob.keys():
            poly2mask(img,blob[label], mask_img_dir, label, classes[label],name)
            classes[label] += 1
                
DATA_DIR = '../data/images'
out_dir = os.path.join(DATA_DIR, 'cropped')
img_files = [img_file for img_file in [files for _, _, files in os.walk(DATA_DIR)][0] if img_file.endswith('.jpg')]
for img in img_files:
    file_name = img.split('.')[0]
    json_file = file_name + '.json'
    img_path = os.path.join(DATA_DIR, img)
    json_path = os.path.join(DATA_DIR, json_file)
    img = cv2.imread(img_path)[:,:,::-1]
    json_to_mask(json_path, out_dir, img)
