import cv2
import numpy as np
import os

DATA_DIR = '../data/'
IMG_FILES = [img_file for img_file in [files for _,_,files in os.walk(DATA_DIR)][0] if img_file.endswith('.jpg')]

image = cv2.imread(os.path.join(DATA_DIR, IMG_FILES[2]))
height = image.shape[0]
width = image.shape[1]
new_height = int(height / 5)
new_width = int(width / 5)
image = cv2.resize(image, (new_width, new_height))
out = image.copy()
print(image.shape)

'''
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
 
# ensure at least some circles were found
if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	circles = np.round(circles[0, :]).astype("int")
 
	# loop over the (x, y) coordinates and radius of the circles
	for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		cv2.circle(out, (x, y), r, (0, 255, 0), 4)
		cv2.rectangle(out, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
 
	# show the output image
	cv2.imshow("output", np.hstack([image, out]))
	cv2.waitKey(0)
'''

