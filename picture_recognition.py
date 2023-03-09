# -*- coding: utf-8 -*-
"""
Spyder Editor

Making a contour to the object in the image

@author: lalepi
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt





"image acqusition and masking, determine the intensity of lower and upper limit"

img = cv2.imread('black.png')

print('Original Dimensions : ',img.shape)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

lower = np.array([0,0,0])
higher = np.array([200,200,200])

mask = cv2.inRange(img,lower,higher)
copy_img = img.copy()

"Contour finding"

cont,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

cont_img = cv2.drawContours(img, cont, -1, 255, 3)

contours = max(cont, key = cv2.contourArea)

x,y,w,h =cv2.boundingRect(contours)

cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)

cropped_image = img[y:y+h,x:x+w]


"show images"

plt.figure(figsize=(20,4))
plt.subplot(1,3,1),plt.imshow(copy_img)
plt.subplot(1,3,2),plt.imshow(img)
plt.subplot(1,3,3),plt.imshow(cropped_image)

