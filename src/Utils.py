"""
This file contains all commonly used functions. They mostly deal 
with using skimage, specific numpy operations, functions to make 
Laplacian and Gaussian stacks, and our warping function.
"""

import cv2
import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
from imutils import face_utils
from Settings import *
from scipy.interpolate import RegularGridInterpolator
from skimage.draw import polygon
from scipy.interpolate import interp2d

def get_facial_landmarks(detector, predictor, image):
    print("# Getting facial landmarks...")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    shape = None

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        shape = predictor(image, rect)
        shape = face_utils.shape_to_np(shape)
        corners = [(0,0), (image.shape[1],0), (0, image.shape[0]), (image.shape[1], image.shape[0])]

        for corner in corners:
            shape = np.vstack((shape, corner))

    return shape

def read_image(file, color=cv2.COLOR_BGR2RGB, img_float=False):
    image = cv2.imread(INPUTS_FOLDER + file + FILETYPE, color)
    
    if img_float:
        return image.astype(np.float32) / 255.0
    
    return image

def show_points(image, points):
    plt.plot(points[:,0], points[:,1], 'o')
    plt.imshow(image)
    plt.show()


def show_triangulation(image, points, tri):
    plt.triplot(points[:,0], points[:,1], tri.simplices)
    plt.plot(points[:,0], points[:,1], 'o')
    plt.imshow(image)
    plt.show()
    

def rescale(img):
    """
    Assures values between 0 and 1
    """
    return (img - img.min()) / (img.max() - img.min())


def lowPass(image, size, sigma):
    """
    Applies a low pass filter to the image given the kernel 
    size and sigma.
    """
    kernel = cv2.getGaussianKernel(size, sigma)
    kernel = np.multiply(kernel, kernel.transpose())
    return cv2.filter2D(image, -1, kernel)


# Return homogenous matrix containing points
def homogenous(points):
    return np.row_stack((points.transpose(), (1, 1, 1)))

def get_residuals(input, example):
    kernel_size = 2 ** STACKS_DEPTH
    return lowPass(input, 5 * kernel_size, kernel_size), lowPass(example, 5 * kernel_size, kernel_size)

"""
Used to recover the affine transformation given two sets of points
tri_points2 = np.dot(tri_points1, R) + t
where R is the rotation portion.

Solution found at https://stackoverflow.com/questions/27546081/
determining-a-homogeneous-affine-transformation-matrix-from-six-points-in-3d-usi
"""
def computeAffine(tri_points1, tri_points2):
    #A is going to be the matrix with tri_points1
    source_matrix = homogenous(tri_points1)
    target_matrix = homogenous(tri_points2)
    T = np.matmul(target_matrix, np.linalg.inv(source_matrix))
    return T

def warp_energy_stack(es, input_shape, input_tri, example_shape):
        print("# Warping energy stack...")
        warped_stack = []
        
        for elem in es:
            warped = warp(elem, example_shape, input_shape, input_tri)
            warped_stack.append(warped)

        return warped_stack

def warp(image, source_points, target_points, tri):
    # Warps IMAGE with SOURCE_POINTS to TARGET_POINTS with TRI
    print("# Warping image...")
    imh, imw = image.shape
    out_image = np.zeros((imh, imw))
    xs = np.arange(imw)
    ys = np.arange(imh)

    interpFN = RegularGridInterpolator((ys, xs), image, bounds_error=False, fill_value=0)

    for triangle_indices in tri.simplices:

        source_triangle = source_points[triangle_indices]
        target_triangle = target_points[triangle_indices]
        A = computeAffine(source_triangle, target_triangle)
        A_inverse = np.linalg.inv(A)

        tri_rows = target_triangle.transpose()[1]
        tri_cols = target_triangle.transpose()[0]

        row_coordinates, col_coordinates = polygon(tri_rows, tri_cols)

        for x, y in zip(col_coordinates, row_coordinates):
            # Point inside target triangle mesh
            point_in_target = np.array([x, y, 1])

            # Point inside source image
            point_on_source = np.dot(A_inverse, point_in_target)

            x_source, y_source = point_on_source[:2]

            # Interpolate source value
            source_value = interpFN([y_source, x_source])  # Note: (y, x) order for RegularGridInterpolator

            try:
                out_image[y, x] = source_value
            except IndexError:
                continue

    return out_image


