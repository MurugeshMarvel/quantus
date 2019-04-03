import os
import json

DATA_DIR = '../data'
ANNOT_EXT = '.json'
IMG_EXT = '.jpg'
IMAGE_DIR = 'images'
ANNOT_DIR = 'annotations'

img_files = [img_file for img_file in [files for  _, _, files in os.walk(DATA_DIR)][0] if img_file.endswith(IMG_EXT)]
json_files = [json_file for json_file in [files for _, _, files in os.walk(DATA_DIR)][0] if json_file.endswith(ANNOT_EXT)]
img_files = [img_file for img_file in img_files if (img_file.split('.')[0]+ANNOT_EXT) in json_files]

if not os.path.isdir(os.path.join(DATA_DIR, IMAGE_DIR)):
    os.system(f"mkdir {os.path.join(DATA_DIR, IMAGE_DIR)}")
if not os.path.isdir(os.path.join(DATA_DIR, ANNOT_DIR)):
    os.system(f"mkdir {os.path.join(DATA_DIR, ANNOT_DIR)}")

for annt_file, img_file in zip(json_files, img_files):
    annt_path = os.path.join(DATA_DIR, annt_file)
    img_path = os.path.join(DATA_DIR, img_file)
    new_annt_path = os.path.join(DATA_DIR, ANNOT_DIR, annt_file)
    new_img_path = os.path.join(DATA_DIR, IMAGE_DIR, img_file)
    os.system(f"cp {annt_path} {new_annt_path}")
    os.system(f"cp {img_path} {new_img_path}")
    print(f"finished copying {annt_file.split('.')[0]}")