import cv2
import numpy as np

#import image
image = cv2.imread(r'.\result\bnw.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#cv2.imshow('orig',image)
#resize = cv2.resize(image,None,fx=1, fy=1, interpolation = cv2.INTER_LINEAR)
#grayscale
img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

img = cv2.bitwise_not(img)
#th2 = cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,-2)
#cv2.imshow("th2", th2)
#cv2.imwrite("th2.jpg", th2)

#binary
ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
#cv2.imshow('second',thresh)
#cv2.waitKey(0)
#thresh = cv2.resize(thresh,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)

#dilation
kernel = np.ones((5,100), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
#cv2.imshow('dilated',img_dilation)
#cv2.waitKey(0)

#find contours
ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi = image[y:y+h, x:x+w]

    # show ROI
    #cv2.imshow('segment no:'+str(i),roi)
    line = cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
    file = r'.\result\line_'+str(i)+'.jpg'
    cv2.imwrite(file,roi)
    #cv2.waitKey(0)

#cv2.imshow('marked areas',image)
#cv2.waitKey(0)