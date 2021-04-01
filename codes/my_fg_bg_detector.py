# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 00:53:44 2021

@author: Xianglun Xu

Email: xianglun918@gmail.com
"""


# Libraries
import cv2
# import collections
import numpy as np

np.set_printoptions(threshold=np.inf)

def my_fg_bg_detector(gray):
    '''
    Use foreground / background difference to get the contour.

    Parameters
    ----------
    img : Input should be a gray image.

    Returns
    -------
    sure_fg : Return the detected area. (Foreground.)
        
    '''
    if len(gray.shape) > 2:
        raise ValueError("Input is not a gray image.")
    
    # gray = cv2.equalizeHist(gray)
    
    # OTSU thresholding
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations=2)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=3)

    # Get sure background area
    # sure_bg = cv2.dilate(closing, kernel, iterations=3)
    sure_bg = closing    
    
    # Get sure foreground area (dist_transform extracts the foreground)
    dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 3)

    # Threshold the dist_transform
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)

    # Reverse and closing to gain the final result
    # sure_fg = np.ones(sure_fg.shape) * 255 - sure_fg 
    sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30)))
    
    # To make sure the image is not inverse, it happens when bg is brigter than fg

    if sure_fg[0][0] == 255:
        sure_fg = np.ones(sure_fg.shape) * 255 - sure_fg
    
    return sure_fg

############ Test code for direct view on effect of each step. #############

if __name__ == '__main__':
    
    roi = cv2.imread("../images/test_imgs/cool_guy_face.jpg")
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # gray = cv2.equalizeHist(gray)
    ret, mask1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3, 3), np.uint8)
    mask2 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    mask3 = cv2.dilate(mask2, kernel, iterations=3)
    
    mask4 = cv2.distanceTransform(mask2, cv2.DIST_L2, 3)
    
    # Threshold the dist_transform
    ret, mask5 = cv2.threshold(mask4, 0.1 * mask4.max(), 255, 0)
    
    while 1:
        cv2.imshow('1', mask1)
        cv2.imshow('2', mask2)
        cv2.imshow('3', mask3)
        cv2.imshow('4', mask4)
        cv2.imshow('5', mask5)
        cv2.imshow('sure_fg', my_fg_bg_detector(gray))
        
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            # cv2.imwrite("canny_test_future.jpg", result)
            break
    cv2.destroyAllWindows()