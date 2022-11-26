# -*- coding: utf-8 -*-
"""
Last modified on Sun Mar 14 16:33:28 2021

@author: xiang

Aggregated codes for all tests and results.
"""
import os
import time
from fc_logger import logger

import cv2
import numpy as np
from tqdm import tqdm

from my_canny_detector import my_canny_detector
from my_fg_bg_detector import my_fg_bg_detector
from my_haar_cascade import my_haar_cascade, obj_detect

# =============================================================================
# Pre-set values
# =============================================================================
SCREEN_W = 1600
SCREEN_H = 900
np.set_printoptions(threshold=np.inf)
FEI_DIR = '../images/FEI_front_faces'
MY_DATASET_DIR = '../images/my_dataset'
HAAR_RESULTS_DIR = '../images/haar_results'
CONTOUR_RESULTS_DIR = '../images/contouring_results'
CONTOUR_COMPARE_DIR = '../images/contouring_compare'


# =============================================================================
#  Some useful self-defined functions
# =============================================================================
def draw_recs(img, locs, color=(0, 0, 255), lengh=2):
    """
    Draw detected rectangles on input image. Limit the size to reduce false alarm.
    """
    # Don't draw if the rec too small
    for (x, y, w, h) in locs:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), color, lengh)

    return None


def img_display(img, w_name="Test", size=(None, None)):
    """Show given image in cv2 window.
    """
    size = img.shape

    mov_w = (SCREEN_W - size[0]) // 2
    mov_h = (SCREEN_H - size[1]) // 2
    if size[0] is not None:
        img = cv2.resize(img, (size[0], size[1]))
    cv2.namedWindow(w_name)
    cv2.moveWindow(w_name, mov_w, mov_h)
    while 1:
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        cv2.imshow(w_name, img)
    cv2.destroyAllWindows()
    return None


# =============================================================================
# Realization of my contouring algorithm
# =============================================================================
def my_contouring(gray):
    contour1 = my_canny_detector(gray)
    contour2 = my_fg_bg_detector(gray)
    contour = contour1 + contour2
    contour[contour > 255] = 255
    contour = cv2.morphologyEx(contour, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))
    return contour


def test_my_contouring(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result1 = my_canny_detector(gray)
    result2 = my_fg_bg_detector(gray)
    result = result1 + result2
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
    img_display(result1)
    img_display(result2)
    img_display(result)


# =============================================================================
# Functions to implement different tests 
# =============================================================================

def test_on_camera(test_img_path=None):
    """Test the algorithm on live frame captured by camera.

    Returns
    -------
    None.
    """

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    cap = cv2.VideoCapture(0)
    face_xml = "./opencv_xml/haarcascade_frontalface_default.xml"
    # face_xml = "./opencv_xml/my15stages.xml"
    # eye_xml = "./opencv_xml/haarcascade_frontalface_default.xml"

    while True:

        if test_img_path == None:
            ret, frame = cap.read()
        else:
            frame = cv2.imread(test_img_path)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization for better performance in low contrast conditions
        # gray = cv2.equalizeHist(gray)

        # Detect faces and Draw
        faces = obj_detect(gray, face_xml)
        draw_recs(frame, faces)

        for (x, y, w, h) in faces:
            # frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            # Get the contour
            contour = my_contouring(roi_gray)

            # Add the contour back
            contour = contour.astype(np.uint8)
            contour = cv2.cvtColor(contour, cv2.COLOR_GRAY2BGR)
            contour[np.where((contour == [255, 255, 255]).all(axis=2))] = [255, 0, 0]

            # frame[y:y + h, x:x + w] = cv2.bitwise_and(contour, roi_color)
            # frame[y:y + h, x:x + w] = 0.2 * contour + 1 * roi_color
            frame[y:y + h, x:x + w] = cv2.addWeighted(contour, 2, roi_color, 1, 0)

        cv2.imshow('frame', frame)
        out.write(frame)
        # esc to exit
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def check_haar_cascade_success_rate():
    '''
    Check the haar_cascade success percentage on FEI dataset.

    Returns
    -------
    None.

    '''
    files = os.listdir(FEI_DIR)
    success = 0
    total = len(files)

    time1 = time.time()

    for img_name in tqdm(files):
        img_path = FEI_DIR + '/' + img_name
        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = my_haar_cascade(gray)
        if len(faces) > 0:
            success += 1
    time2 = time.time()
    logger.info('\nTime Collapse: %s sec' % round(time2 - time1, 2))
    logger.info('The success rate is: %s%%' % (success / total * 100))


def do_classification():
    """Perform haar-cascade on my dataset and store the results.

    Returns
    -------
    None.
    """
    logger.info('Performing classification...')
    files = os.listdir(MY_DATASET_DIR)
    save_path = HAAR_RESULTS_DIR

    time1 = time.time()

    for img_name in files:
        img_path = MY_DATASET_DIR + '/' + img_name
        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = my_haar_cascade(gray)
        for i in range(len(faces)):
            face = faces[i]
            [x, y, w, h] = face
            roi = image[y:y + h, x:x + w]
            cv2.imwrite(save_path + '/' + img_name[:-5] + '_face#%s.jpg' % i, roi)

    time2 = time.time()
    logger.info('Time Collapse: %s sec' % round(time2 - time1, 2))
    logger.info('Faces saved to: ' + save_path)


def do_contouring():
    """Do contour action to my dataset and save the result images.

    Returns
    -------
    None.
    """
    logger.info('Performing contouring...')

    files = os.listdir(HAAR_RESULTS_DIR)
    save_path = CONTOUR_RESULTS_DIR

    time1 = time.time()

    for img_name in files:
        img_path = HAAR_RESULTS_DIR + '/' + img_name
        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contour = my_contouring(gray)
        cv2.imwrite(save_path + '/' + img_name[:-4] + '_contour.jpg', contour)

    time2 = time.time()
    logger.info('Time Collapse: %s sec' % round(time2 - time1, 2))
    logger.info('Contours saved to: ' + save_path)


# =============================================================================
# The test part
# =============================================================================

if __name__ == "__main__":
    logger.info('Running...')
    # =============================================================================
    #     Uncomment below lines to perform each task.
    # =============================================================================
    test_on_camera()
    # test_my_contouring('../images/test_imgs/cool_guy_face.jpg')
    # check_haar_cascade_success_rate()
    # do_classification()
    # do_contouring()
