# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 20:04:40 2021

@author: Xianglun Xu

Email: xianglun918@gmail.com
"""

# Libraries
import cv2
import collections
import numpy as np

# Pre-defined parameters 
MIN_VAL = 100
MAX_VAL = 300

# Perform Canny Edge detection
def my_canny_detector(gray):
    '''
    Apply Canny edge detection.

    Parameters
    ----------
    gray : Input should be a gray image.

    Returns
    -------
    result : Return the detected area.

    '''
    # Do the Canny edge detection
    if len(gray.shape) > 2:
        raise ValueError("Input is not a gray-scale image.")
    
    # gray = cv2.equalizeHist(gray)
    
    # Binarization 
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Do a closing to eliminate small holes on edges.
    result = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, 
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
    # Do Canny edge detection
    result = cv2.Canny(result, MIN_VAL, MAX_VAL)
    # Blur the edges to make it wider
    result = cv2.GaussianBlur(result, (3, 3), 0)
    # Do Canny edge detection agin (Now there would be double lines)
    result = cv2.Canny(result, MIN_VAL, MAX_VAL)
    # Forge double lines to the solid fully connected line.
    # result = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, 
    #                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
    
    # Finally, fill the detected area - use bfs technique
    rn, cn = len(result), len(result[0])
    r, c = rn // 2, cn // 2   # Start from center
    result[r][c] = 255
    neighbors = collections.deque([(r, c)])
    # print(result.shape)
    while neighbors:
        row, col = neighbors.popleft()
        for x, y in [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]:
            if  (0 <= x < rn and 0 <= y < cn and result[x][y] != 255):
                # print(x, y)
                neighbors.append((x, y))
                result[x][y] = 255
                
    # We observe that there are still eyes', and nose's part are not closed
    # So apply closing again.
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))
    
    ## Uncommon the below line to get the edge contour
    ## result = cv2.GaussianBlur(result, (3, 3), 0) - result
    
    return result


############ Test code for direct view on effect of each step. #############

if __name__ == '__main__':
    
    roi = cv2.imread("../images/test_imgs/cool_guy_face.jpg")
    # roi = cv2.resize(roi, (360, 260))
   
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # gray = cv2.equalizeHist(gray)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    mask1 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
    mask2 = cv2.Canny(mask1, 100, 300)
    # mask2 = mask1
    mask3 = cv2.GaussianBlur(mask2, (3, 3), 0)
    mask4 = cv2.Canny(mask3, 100, 300)
    
    mask5 = cv2.morphologyEx(mask4, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))

    mask4_5 = mask5.copy()
    
    rn, cn = len(mask5), len(mask5[0])
    r = len(mask5) // 2
    c = len(mask5[0]) // 2
    neighbors = collections.deque([(r, c)])
    while neighbors:
        row, col = neighbors.popleft()
        for x, y in [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]:
            if  0 <= x < rn and 0 <= y < cn and mask5[x][y] != 255:
                neighbors.append((x, y))
                mask5[x][y] = 255
    mask6 = cv2.morphologyEx(mask5, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))
    
    while 1:
        cv2.imshow('1', mask1)
        cv2.imshow('2', mask2)
        cv2.imshow('3', mask3)
        cv2.imshow('4', mask4)
        cv2.imshow('5', mask5)
        cv2.imshow('6', mask6)
        cv2.imshow('4-5', mask4_5)
        cv2.imshow('result', my_canny_detector(gray))
        # cv2.imshow('1', bound_contours(roi)[0])
        
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            # cv2.imwrite("canny_test_future.jpg", result)
            break
    cv2.destroyAllWindows()