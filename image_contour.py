# -*- coding: utf-8 -*-
"""
Spyder Editor

Making a contour to the object in the image
Locating the centerpoint of the wanted contour

@author: lalepi
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



img_counter = 1

# cam = cv2.VideoCapture(0)

# cv2.namedWindow("test")



# while True:
#     ret, frame = cam.read()
#     if not ret:
#         print("failed to grab frame")
#         break
#     cv2.imshow("test", frame)

#     k = cv2.waitKey(1)
#     if k%256 == 27:
#         # ESC pressed
#         print("Escape hit, closing...")
#         break
#     elif k%256 == 32:
#         # SPACE pressed
#         img_name = "opencv_frame_{}.png".format(img_counter)
#         cv2.imwrite("material/opencv_images/" + img_name, frame)
#         print("{} written!".format(img_name)) 
#         img_counter += 1

# cam.release()

# cv2.destroyAllWindows()

img_name = "opencv_frame_{}.png".format(img_counter)

path = 'material/test_images/'

"Read image"
img = cv2.imread(path + img_name)
image_copy = img.copy()


"Grayscaling & threshold"
gray_scale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#blurred = cv2.GaussianBlur(gray_scale,(5,5),cv2.BORDER_DEFAULT)

(T, threshInv) = cv2.threshold(gray_scale,230,250,cv2.THRESH_BINARY_INV)

blurred = cv2.GaussianBlur(threshInv,(5,5),cv2.BORDER_CONSTANT)
"Create Contours"
cont,_ = cv2.findContours(blurred, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours = sorted(cont, key=cv2.contourArea, reverse=True)
contour_image = cv2.drawContours(image_copy, contours, -1,(0,200,0), 3);

"Use Minimal contour area"
rect = cv2.minAreaRect(contours[2])
box = cv2.boxPoints(rect)

"retview moments for center point location"
M = cv2.moments(box)
area = 'm00' # area of the contour
x_val = 'm10' # x-axis of the contour
y_val = 'm01' # y-axis of the contour

"Calculate the X-coordinate  & Y-coordinate of the centroid"
"If blob area is empty, ignore"
if M[area] != 0:
    cx = int(M[x_val] / M[area])
    cy = int(M[y_val] / M[area])
else:
    cx,Cy = 0, 0

"draw circle to the centerpoint of the contour"
cv2.circle(image_copy, (cx,cy),10,(0,0,255),-1)


"Outputs"


"actual image"
# plt.figure(figsize=[10,10])
# plt.imshow(image_copy[:,:,::-1]); #plt.axis("off")

"centervalue"
print ('Centroid:({},{})'.format(cx,cy))

"Moments"
print(M)

print('cx value is:', cx)
print('cy value is:', cy)

"test windows"
print('Original Dimensions : ',img.shape)
# plt.figure(figsize=[20, 4])
# plt.subplot(1, 3, 1), plt.imshow(gray_scale) 
# plt.subplot(1, 3, 2), plt.imshow(threshInv)
# plt.subplot(1, 3, 3), plt.imshow(image_copy[:,:,::-1])

fig = plt.figure(figsize=[15, 15])

plt.subplot(2, 2, 1), plt.imshow(gray_scale,cmap="gray", vmin=0, vmax=255)
plt.title("GrayScale")

plt.subplot(2, 2, 2), plt.imshow(threshInv)
plt.title("Threshold")

plt.subplot(2, 2, 3), plt.imshow(blurred,cmap="gray", vmin=0, vmax=255)
plt.title("Blurred")

plt.subplot(2, 2, 4), plt.imshow(image_copy[:,:,::-1])
plt.title("Contour Image")


#Load the calibration parameters

with np.load('CameraParameters.npz') as file:
    mtx, dist, rvecs, tvecs = [file[i] for i in('cameraMatrix','dist','rvecs','tvecs')]
    
img = cv2.imread(path + img_name)
h,  w = img.shape[:2]
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))


# Undistort
dst = cv2.undistort(img, mtx, dist, None, newCameraMatrix)


cv2.imwrite('calibrated_'+img_name, dst)















