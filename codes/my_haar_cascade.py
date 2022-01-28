# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 01:21:21 2021

@author: Xianglun Xu

Email: xianglun918@gmail.com
"""

import cv2
# import numpy as np
from matplotlib import pyplot as plt
import imutils

# Some pre-defined parameters
FACE_XML = "./opencv_xml/haarcascade_frontalface_default.xml"

def obj_detect(img, xml, scale=1.1, nei=3):
    """
    Return Haar-Cascade classifier results. (x, y, w, h) - square shape
    """
    obj_cascade = cv2.CascadeClassifier(xml)
    objects = obj_cascade.detectMultiScale(img, scale, nei)
    return objects

def my_haar_cascade(gray):
    '''
    Perform Haar-Cascade classifier on input image.

    Parameters
    ----------
    gray : Input should be a gray image.

    Returns
    -------
    faces : Detected regions of faces. In the form of [(x, y, w, h), ...].

    '''
    faces = obj_detect(gray, FACE_XML)
    ret = []
    for (x, y, w, h) in faces:
        if w * h > 30000:
            ret.append([x-50, y-70, w+80, h+100])
            
    return ret

if __name__ == '__main__':
    
    img = cv2.imread('../images/test_imgs/cool_guy.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = my_haar_cascade(gray)
    (x, y, w, h) = result[0]
    roi = img[y:y + h, x:x + w]
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(imutils.opencv2matplotlib(img))
    plt.title('Original')
    plt.subplot(1, 2, 2)
    plt.imshow(imutils.opencv2matplotlib(roi))
    plt.title('Detected')
    
    print(result)
    while 1:
        
        cv2.imshow('Result', roi)
        
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            # cv2.imwrite("canny_test_future.jpg", result)
            break
    cv2.destroyAllWindows()
