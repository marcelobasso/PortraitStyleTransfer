"""
This file contains all commonly used functions. They mostly deal 
with using skimage, specific numpy operations, functions to make 
Laplacian and Gaussian stacks, and our warping function.
"""

import skimage.io as skio
import skimage as sk
from imutils import face_utils
from Settings import *
import cv2
import numpy as np

def get_facial_landmarks(detector, predictor, image):
    print("# Getting facial landmarks...")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    # plt.figure()
    triangulation = None
    shape = None

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(image, rect)
        shape = face_utils.shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        # (x, y, w, h) = face_utils.rect_to_bb(rect)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the face number
        # cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
        # 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        # for (x, y) in shape:
        #     plt.scatter(x, y, s=10)

        corners = [(0,0), (image.shape[1],0), (0, image.shape[0]), (image.shape[1], image.shape[0])]

        for corner in corners:
            shape = np.vstack((shape, corner))

    return shape

def read_image(file, color=cv2.COLOR_BGR2RGB, img_float=False):
    image = cv2.imread(INPUTS_FOLDER + file + FILETYPE, color)
    
    if img_float:
        return image.astype(np.float32) / 255.0
    
    return image