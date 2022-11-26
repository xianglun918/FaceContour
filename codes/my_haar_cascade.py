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
from fc_logger import logger

# Use pre-trained model defined in .xml file.
FACE_XML = "./opencv_xml/haarcascade_frontalface_default.xml"


def obj_detect(img_val, xml, scale=1.1, nei=3):
    """Return Haar-Cascade classifier results. (x, y, w, h) - square shape
    """
    obj_cascade = cv2.CascadeClassifier(xml)
    objects = obj_cascade.detectMultiScale(img_val, scale, nei)
    return objects


def my_haar_cascade(gray_img):
    """Perform Haar-Cascade classifier on input image.

    Parameters
    ----------
    gray_img : Input should be a gray image.

    Returns
    -------
    faces : Detected regions of faces. In the form of [(x, y, w, h), ...].
    """
    faces = obj_detect(gray_img, FACE_XML)
    ret = []
    for (x_, y_, w_, h_) in faces:
        if w_ * h_ > 30000:
            ret.append([x_ - 50, y_ - 70, w_ + 80, h_ + 100])

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

    logger.info(result)
    while 1:
        cv2.imshow('Result', roi)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            # cv2.imwrite("canny_test_future.jpg", result)
            break
    cv2.destroyAllWindows()
