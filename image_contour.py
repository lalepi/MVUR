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


"Read image"
img = cv2.imread('color.png')
image_copy = img.copy()


"Grayscaling & threshold"
gray_scale = cv2.cvtColor(image_copy,cv2.COLOR_BGR2GRAY)
(T, threshInv) = cv2.threshold(gray_scale,200,255,cv2.THRESH_BINARY_INV)                        
                        
"Create Contours"
cont,_ = cv2.findContours(threshInv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours = sorted(cont, key=cv2.contourArea, reverse=True)
contour_image = cv2.drawContours(image_copy, contours, -1,(0,255,0), 3);
"Use Minimal contour area"
rect = cv2.minAreaRect(contours[0])
box = cv2.boxPoints(rect)

"retviewe moments for center point location"
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
    cX,Cy = 0, 0

"draw circle to the centerpoint of the contour"
cv2.circle(image_copy, (cx,cy),10,(0,0,255),-1)







"Outputs"



"test windows"
print('Original Dimensions : ',img.shape)
#plt.figure(figsize=[20, 4])
#plt.subplot(1, 3, 1), plt.imshow(gray_scale)
#plt.subplot(1, 3, 2), plt.imshow(threshInv)
#plt.subplot(1, 3, 3), plt.imshow(threshInv)



"actual image"
plt.figure(figsize=[10,10])
plt.imshow(image_copy[:,:,::-1]); #plt.axis("off")

"centervalue"
print ('Centroid:({},{})'.format(cx,cy))

"Moments"
print(M)


