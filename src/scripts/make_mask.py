"""
This file opens up an image and allows you 
to draw and save a binary mask. The code 
written in this file is not our own. 
The instructions to use this code is found at:
https://github.com/nikhilushinde/cs194-26_proj3_2.2

Args:
    Entire path of image to draw mask around

Example: 
    $ python3 ./make_mask.py <image>.jpg
"""

import cv2
import sys
import numpy as np

# global variables for drawing on mask
pressed_key = 0
drawing = False
polygon = False
centerMode = False
contours = []
polygon_center = None
img = None

path = sys.argv[1]
output = path.split('.')[0] + '_mask.jpg' # no extension
masks_to_ret = {"centers":[], "contours":[], "offsets":[]}

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global drawing, centerMode, polygon, pressed_key
    if drawing == True and event == cv2.EVENT_MOUSEMOVE:
        cv2.circle(img,(x,y),10,(255,255,255),-1)
        cv2.circle(mask,(x,y),10,(255,255,255),-1)

    if polygon == True and event == cv2.EVENT_LBUTTONDOWN:
        contours.append([x,y])
        cv2.circle(img,(x,y),2,(255,255,255),-1)
    
    if centerMode == True and event == cv2.EVENT_LBUTTONDOWN:
        polygon_center = (x,y)
        print(polygon_center)
        cv2.circle(img, polygon_center, 3, (255, 0, 0), -1)
        centerMode = False

        masks_to_ret["centers"].append(polygon_center)
        masks_to_ret["contours"].append(contours)

# Create a black image, a window and bind the function to window
orig_img = cv2.imread(path)
reset_orig_img = orig_img[:]
mask = np.zeros(orig_img.shape, np.uint8)
img = np.array(orig_img[:])
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600, 600)
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    pressed_key = cv2.waitKey(20) & 0xFF

    """
    Commands:
    d: toggle drawing mode
    p: toggle polygon mode
    q: draw polygon once selected, and select center
    """

    if pressed_key == 27:
        break
    elif pressed_key == ord('p'):
        polygon = not polygon
    elif polygon == True and pressed_key == ord('q') and len(contours) > 2:
        contours = np.array(contours)
        cv2.fillPoly(img, pts=[contours], color = (255,255,255))
        cv2.fillPoly(mask, pts=[contours], color = (255,255,255))
        print("Saving mask to " + output)
        cv2.imwrite(output, mask)