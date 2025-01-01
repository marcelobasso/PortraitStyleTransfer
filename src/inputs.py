"""
This file opens up each image, using ginput and 
allows to click and select locations for 
correspondence points.

Argus:
    First image to choose points on without extension.
    Second image to choose points on without extension
    Number of points to be selected on each image.

Example: 
    $ python3 ./inputs.py jose george 43
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import skimage.io as skio
from scipy.spatial import Delaunay

imname = sys.argv[1]
im2name = sys.argv[2]
num_points = int(sys.argv[3])
image1 = skio.imread('./assets/inputs/' + imname + '.jpg')
image1_full = sk.img_as_float(image1)

image2 = skio.imread('./assets/inputs/' + im2name + '.jpg')
image2_full = sk.img_as_float(image2)

plt.imshow(image1_full)
print("Please click")
A_points = plt.ginput(n=num_points, timeout=500, show_clicks=True)
A_corres = np.array(A_points)
np.savetxt('./points/' + imname + '_points.txt', A_corres, fmt='%f')
plt.close()
print("clicked", A_corres)
plt.imshow(image2_full)
B_corres = np.array(plt.ginput(n=num_points, timeout=500, show_clicks=True))
np.savetxt('B_points_' + im2name + '.txt', B_corres, fmt='%f')
plt.close()
print('clicked', B_corres)
