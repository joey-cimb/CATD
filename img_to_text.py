# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 19:13:47 2019

@author: mycp2cqj
"""

from PIL import Image
import pytesseract
import cv2
import numpy as np

im = Image.open('C:\\Users\\mycp2cqj\\Documents\\CATD_PROJECT\\picture.png')
im.save('C:\\Users\\mycp2cqj\\Documents\\CATD_PROJECT\\sample.tiff')
filename="C:\\Users\\mycp2cqj\\Documents\\CATD_PROJECT\\sample.tiff"
img_raw2 = cv2.imread("C:\\Users\\mycp2cqj\\Documents\\CATD_PROJECT\\picture.png")
cv2.imshow('Original', img_raw2)

img_raw = cv2.fastNlMeansDenoisingColored(img_raw2,None,10,10,7,21)

#resize
scale_percent = 105 # percent of original size
width = int(img_raw.shape[1] * scale_percent / 100)
height = int(img_raw.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
img = cv2.resize(img_raw, dim, interpolation = cv2.INTER_AREA)

# Create our shapening kernel, it must equal to one eventually
kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])
# applying the sharpening kernel to the input image & displaying it.
sharpened = cv2.filter2D(img, -4, kernel_sharpening)
#-4
kernel = np.ones((1,1),np.uint8)
erosion = cv2.erode(sharpened,kernel,iterations = 1)
dilation = cv2.dilate(sharpened,kernel,iterations = 1)
#cv2.imwrite(filename,sharpened)
#blur = cv2.GaussianBlur(sharpened,(1,1),0)
smooth = cv2.addWeighted(dilation,1.0,img,0,0)
smooth = cv2.fastNlMeansDenoisingColored(smooth,None,10,10,7,21)
option = smooth
ret,thresh1 = cv2.threshold(option,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(option,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(option,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(option,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(option,127,255,cv2.THRESH_TOZERO_INV)
res = cv2.resize(thresh4,None,fx=3, fy=3, interpolation = cv2.INTER_LINEAR)

#plt.imshow(resized)
#fxfy 5
#thresh3
#cv2.imwrite(filename,sharpened)
cv2.imwrite(filename,res)

pytesseract.pytesseract.tesseract_cmd="C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
text=pytesseract.image_to_string(Image.open(filename),lang='eng')
print(text)
#plt.imshow(res)
#plt.imshow(img)
#plt.imshow(sharpened)
