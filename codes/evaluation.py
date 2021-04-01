# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 18:15:07 2021

@author: Xianglun Xu

Email: xianglun918@gmail.com
"""

from main import *
import matplotlib.pyplot as plt
import imutils
import collections
from PIL import Image
# from PIL import Image

# Pre-set values
TEST_IMG_PATH = '../images/test_imgs/cool_guy_face.jpg'
SAVE_PATH = '../images/figures'
CONTOUR_RESULTS_DIR = '../images/contouring_results'
CONTOUR_COMPARE_DIR = '../images/contouring_compare'

# =============================================================================
#  Draw the figures used in report. (Actually went through each realization agian.)
# =============================================================================
def draw_fg_bg_process():
    '''
    Draw the foreground / background contouring process

    Returns
    -------
    None.

    '''
    
    
    plt.figure('Foreground/Background Method Contouring')
    plt.suptitle('Foreground/Background Method Contouring')
    
    # original image
    test_img = cv2.imread(TEST_IMG_PATH)
    plt.subplot(2, 3, 1)
    plt.title('Original')
    plt.imshow(imutils.opencv2matplotlib(test_img))
    
    # gray image
    gray = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
    plt.subplot(2, 3, 2)
    plt.title("Gray-scale")
    plt.imshow(gray, cmap='gray')
    
    # OTSU thresholding
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    plt.subplot(2, 3, 3)
    plt.title('OTSU\'s threshold')
    plt.imshow(thresh, cmap='gray')
    
    # closing
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    sure_bg = closing    
    
    plt.subplot(2, 3, 4)
    plt.title('Closing')
    plt.imshow(sure_bg, cmap='gray')
    
    # distance transform
    dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 3)
    plt.subplot(2, 3, 5)
    plt.title('Distance transform')
    plt.imshow(dist_transform, cmap='gray')
    
    # final result, threshold again
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30)))
    plt.subplot(2, 3, 6)
    plt.title('Final Result')
    plt.imshow(sure_fg, cmap='gray')
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=.5, hspace=.5)
    plt.savefig(SAVE_PATH + '/fg_bg_process.jpg') 

def draw_canny_process():
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=.5, hspace=.5)
    plt.figure('Canny Method Contouring')
    plt.suptitle('Canny Method Contouring')
    
    # original image
    test_img = cv2.imread(TEST_IMG_PATH)
    plt.subplot(2, 3, 1)
    plt.title('Original')
    plt.imshow(imutils.opencv2matplotlib(test_img))
    
    # Same three steps as the fg_bg method
    gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    result = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, 
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
    plt.subplot(2, 3, 2)
    plt.title('Denoise 3 steps\' result')
    plt.imshow(result, cmap='gray')
    
    # First Canny Edge detection
    result = cv2.Canny(result, 100, 300)
    plt.subplot(2, 3, 3)
    plt.title('1st Canny edge')
    plt.imshow(result, cmap='gray')
    
    # Gaussian Blure
    result = cv2.GaussianBlur(result, (3, 3), 0)
    plt.subplot(2, 3, 4)
    plt.title('Gaussian Blur')
    plt.imshow(result, cmap='gray')
    
    # Second Canny
    result = cv2.Canny(result, 100, 300)
    plt.subplot(2, 3, 5)
    plt.title('2nd Canny edge')
    plt.imshow(result, cmap='gray')
    
    # Fill in the blank & do closing again
    rn, cn = len(result), len(result[0])
    r, c = rn // 2, cn // 2   # Start from center
    result[r][c] = 255
    neighbors = collections.deque([(r, c)])
    while neighbors:
        row, col = neighbors.popleft()
        for x, y in [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]:
            if  0 <= x < rn and 0 <= y < cn and result[x][y] != 255:
                # print(x, y)
                neighbors.append((x, y))
                result[x][y] = 255
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))
    
    plt.subplot(2, 3, 6)
    plt.title('Fill the blank')
    plt.imshow(result, cmap='gray')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=.5, hspace=.5)
    
    plt.savefig(SAVE_PATH + '/canny_process.jpg')



def calculate_IOU(template, contour):
    '''
    Calculate the IOU of given two inputs.

    Parameters
    ----------
    template : Gray-scale image, used as reference.
    contour : Gray-scale image, resulted from contouring.

    Returns
    -------
    IOU : Float number indicates the overlapping performance of two areas.

    '''
    if len(template.shape) > 2 or len(contour.shape) > 2:
        raise ValueError("Input not 2D gray-scale. Check conversion.")
    # Technically both template and contour should be binary
    
    # Make sure binarization
    template[template != 255] = 0
    template[template == 255] = 1
    
    contour[contour != 255] = 0
    contour[contour == 255] = 1
    
    
    intersection = np.bitwise_and(template, contour)
    union = np.bitwise_or(template, contour)
    
    IOU = np.sum(intersection==1) / np.sum(union==1)
    
    # Keep 2 decimal places
    IOU = round(IOU, 3)
    return IOU

def evaluate_IOU():
    '''
    Evaluate the IOUs of the result contourings.

    Returns
    -------
    IOU_list : List of IOUs calculated.

    '''
    
    IOU_list = []
    result_files = os.listdir(CONTOUR_RESULTS_DIR)
    compare_files = os.listdir(CONTOUR_COMPARE_DIR)
    for i in range(len(result_files)):
        # print(result_files[i], compare_files[i])
        result_path = CONTOUR_RESULTS_DIR + '/' + result_files[i]
        compare_path = CONTOUR_COMPARE_DIR + '/' + compare_files[i]
        
        result = cv2.imread(result_path)
        compare = cv2.imread(compare_path)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        compare = cv2.cvtColor(compare, cv2.COLOR_BGR2GRAY)
        IOU = calculate_IOU(compare, result)
        IOU_list.append(IOU)
        
        print('Filename: ', compare_files[i], ' IOU: ', IOU)
    print('Average IOU: ', round(sum(IOU_list) / 10, 3))
    
    # Plot the result 
    labels = [i for i in range(1, 11)]
    fig, ax = plt.subplots()
    rects = ax.bar(labels, IOU_list)
    for rect in rects:
        ax.text(rect.get_x(), rect.get_height(), rect.get_height(), ha='left', va='bottom')
    ax.set_ylabel('Percentage')
    ax.set_xlabel('Image #')
    ax.set_title('IOU of each image')
    ax.set_xticks(labels)
    ax.set_xticklabels(labels)
    plt.savefig(SAVE_PATH + '/IOU_result.jpg')

# =============================================================================
# For exhibit contouring result
# =============================================================================

def contour_example():
    img = cv2.imread('../images/test_imgs/cool_guy_face.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contour1 = my_fg_bg_detector(gray)
    contour2 = my_canny_detector(gray)
    contour = contour1 + contour2
    contour = cv2.morphologyEx(contour, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
    contour[contour > 255] = 255
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(contour1, cmap='gray')
    plt.title('Method 1')
    plt.subplot(1, 3, 2)
    plt.imshow(contour2, cmap='gray')
    plt.title('Method 2')
    plt.subplot(1, 3, 3)
    plt.imshow(contour, cmap='gray')
    plt.title('Sum & Closing')
    
    contour = cv2.cvtColor(contour, cv2.COLOR_GRAY2BGR)
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(imutils.opencv2matplotlib(img))
    plt.title('Original')
    
    plt.subplot(1, 2, 2)
    
    img[np.where((contour == [0, 0, 0]).all(axis=2))] = [0, 0, 0]
    plt.imshow(imutils.opencv2matplotlib(img))
    plt.title('Extracted Contour')

def template_example():
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=.5, hspace=.8)
    
    template = '../images/contouring_compare/6-1_face#0.jpg' # 4-1
    original = '../images/haar_results/6-1_face#0.jpg'
    
    img = cv2.imread(original)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    contour1 = my_fg_bg_detector(gray)
    contour2 = my_canny_detector(gray)
    contour = contour1 + contour2
    contour = cv2.morphologyEx(contour, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
    contour[contour > 255] = 255
    
    plt.figure()
    template = Image.open(template)
    original = Image.open(original)
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title('Original')
    
    plt.subplot(1, 3, 2)
    plt.imshow(template, cmap='gray')
    plt.title('Groundtruth Mask')
    
    plt.subplot(1, 3, 3)
    plt.imshow(contour, cmap='gray')
    plt.title('Mask Extracted')

if __name__ == '__main__':
    print('Running...')
    # draw_fg_bg_process()
    # draw_canny_process()
    # evaluate_IOU()
    # contour_example()
    template_example()
    
    
    

