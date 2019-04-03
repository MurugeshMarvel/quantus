import os
from shutil import copyfile
from sklearn.model_selection import train_test_split
file_name_list = []
image_dir = 'images'

file_name_list = [img_file.split('.')[0] for img_file in [files for _, _, files in os.walk(image_dir)][0] if img_file.endswith('.jpg')]
train_files, test_files = train_test_split(file_name_list, test_size = 0.20)
with open("train.txt", "w") as f:
    for file_name in train_files:
        f.write(str(file_name) + "\n")
with open("test.txt",'w') as f:
    for file_name in test_files:
        f.write(str(file_name) + "\n")
print ('Done Copying')