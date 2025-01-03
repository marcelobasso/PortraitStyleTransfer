import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio
import cv2
import scipy.ndimage.interpolation
import scipy.spatial
import scipy.interpolate
from skimage.draw import polygon
from scipy.interpolate import RegularGridInterpolator

#Shows points on a given face image
def showPoints(image, points):
    plt.plot(points[:,0], points[:,1], 'o')
    plt.imshow(image)
    plt.show()


def showImage(image):
    skio.imshow(image)
    skio.show()

def showTri(image, points, tri):
    plt.triplot(points[:,0], points[:,1], tri.simplices)
    plt.plot(points[:,0], points[:,1], 'o')
    plt.imshow(image)
    plt.show()

# Shows a Delaunay triagulation over an image
def showTriSave(image, points, tri, name):
    figure = plt.figure()
    plt.triplot(points[:,0], points[:,1], tri.simplices)
    plt.plot(points[:,0], points[:,1], 'o')
    plt.imshow(image)
    plt.show()
    figure.savefig(name)

# Return homogenous matrix containing points
def homogenous(points):
    return np.row_stack((points.transpose(), (1, 1, 1)))


#Takes a homogenous matrix and recovers the points
def homo_to_points(homo_matrix):
    values = homo_matrix[:2]
    print(values.transpose())

def np_clip(img):
    return np.clip(img, 0, 1)

#Rescaels image between 0 and 1
def rescale(img):
    return (img - img.min()) / (img.max() - img.min())


def L2Norm(image):
    np.linalg.norm(image)

def square_root(matrix):
    return np.sqrt(matrix)

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


# Linear interpolations
def interp(fraction, item1, item2):
    return ((1-fraction) * item1) + (fraction * item2)

#Low pass of a given IMAGE with kernel SIZE and SIGMA
def lowPass(image, size, sigma):
    kernel = cv2.getGaussianKernel(size, sigma)
    kernel = np.multiply(kernel, kernel.transpose())
    return cv2.filter2D(image, -1, kernel)


#Gaussian stack of IMAGE, with DIM_FACTOR the size of the Gaussian kernel
def GaussianStack(image, dim_factor, sigma, stack_depth):
    stack = []
    for i in range(stack_depth + 1):
        image = lowPass(image, dim_factor, sigma)
        sigma *= 2
        stack.append(image)
    return stack

#Laplacian stack given a Gaussian stack
def LaplacianStack(image, stack):
    lap_stack = []
    lap_stack.append(rescale(image - stack[0]))
    for i in range(1, len(stack) - 1):
        lap = stack[i] - stack[i+1]
        lap = rescale(lap)
        lap_stack.append(lap)
    return lap_stack

"""
This functions works to redefine low passing, only this time with a mask.
"""
def lowPassMask(image, size, sigma, mask):
    mask_G = lowPass(mask, size, sigma)
    output = lowPass(image*mask, size, sigma)
    #showImage(output)
    output = rescale(output / mask_G)
    #showImage(output)
    return output

"""
This method is based specifically on the matlab version of the paper.
One thing it does is increase the size of the Gaussian kernel at each level
Likewise, I try to match the implementation as much as possible.
This version also takes in a mask but haven't added that part yet.
Read section 'Using a Mask' in the paper for why this is important.
"""
def LaplacianStackAlt(image, mask, stack_depth, useMask):
    stack = []
    stack.append(image)
    for i in range(1, stack_depth):
        sigma = 2 ** i
        stack.append(lowPass(image, sigma*5, sigma))

    for i in range(len(stack)-1):
        if useMask:
            stack[i] = rescale(stack[i] - stack[i+1]*mask)
        else:
            stack[i] = rescale(stack[i] - stack[i+1])
  
    return stack

# Aggregates all images in STACK
def sumStack(stack):
    final_image = stack[0]
    for i in range(1, len(stack)):
        final_image += stack[i]
  
    return rescale(final_image)

# Warps IMAGE with SOURCE_POINTS to TARGET_POINTS with TRI
def warp(image, source_points, target_points, tri):
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
