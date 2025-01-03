"""
This file contains out algorithm for style transfer. 
Time of execution is approximately 4 minutes with 
image being of size 602x750.

Args:
    Input image, the image to have style transfer to it. Do not include the file extension, please put this image in images/ folder.
    Example image, the image to transfer style from. Do not include the file extension, please put this image in images/ folder.
    Bool: result in grayscale.
    Bool: binary mask in the Laplacian stacks
    Bool: correspondences found manually
    Name of file to be outputted. Do not include extension.

Example: 
    $ python3 ./main.py jose george false true false jose_george_test
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
import skimage.io as skio
import cv2
import skimage.transform as sktr
from imutils import face_utils
import dlib
from functions import *
from Settings import *
import Utils as Utils

# FIX:
# - The output is not being saved correctly
# - Remove multiple variables for input and for masks 
#       Why do I need to load using cv and using skio?
# - define pattern for naming variables and methods

class StyleTransfer():
    def __init__(self, input_file, example_file, gray_img, use_mask, output_file):
        """
        Args:
            input_file (str): name of the input image file
            example_file (str): name of the example image file
            gray_img (bool): flag to indicate if the output should be in grayscale
            use_mask (bool): flag to indicate if the mask should be used in the Laplacian stacks
            output_file (str): name of the output file
        """

        # sets options
        self.input_file = input_file
        self.example_file = example_file
        self.gray_img = gray_img
        self.use_mask = use_mask
        self.output_file = output_file

        # loads images
        self.input = Utils.read_image(input_file)
        self.example = Utils.read_image(example_file)

        # values in the range [0, 1]. But why?
        self.input_f = self.input.astype(np.float32) / 255.0
        self.example_f = self.example.astype(np.float32) / 255.0

        self.input_channels = self.input_f.transpose(2, 0, 1)
        self.example_channels = self.example_f.transpose(2, 0, 1)
        self.input_mask = Utils.read_image(input_file + MASK_SUFIX, color=cv2.IMREAD_GRAYSCALE, img_float=True)
        self.example_mask = Utils.read_image(example_file + MASK_SUFIX, color=cv2.IMREAD_GRAYSCALE, img_float=True)

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('src/shape_predictor_68_face_landmarks.dat')
        self.input_lm = Utils.get_facial_landmarks(self.detector, self.predictor, self.input)
        self.example_lm = Utils.get_facial_landmarks(self.detector, self.predictor, self.example)
        self.output = []

    def run(self):
        """
        Runs style transfer algorithm for each channel of the images.
        In case of gray images, only runs for one channel.
        """

        background = self.extract_background()

        if self.gray_img:
            self.input = sk.color.rgb2gray(self.input_f)
            self.example = sk.color.rgb2gray(self.example_f)
            
            gray = self.transfer_style(self.input, self.example, self.use_mask)
            gray = (sk.color.rgb2gray(background) * (1 - self.input_mask)) + (gray * self.input_mask)
            self.output = gray
        else:
            background_colors = background.transpose(2, 0, 1)
            channels = []

            for c in range(len(self.example_channels)):
                print(f"# Running for color channel {c}...")
                channels.append(self.transfer_style(self.input_channels[c], self.example_channels[c], self.use_mask))
                self.output.append((background_colors[c] * (1 - self.input_mask)) + (channels[c] * self.input_mask))

            self.output = np.dstack(self.output)

        cv2.imwrite(OUTPUTS_FOLDER + self.output_file + FILETYPE, self.output)
        cv2.imshow('result', self.output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # This is based more off of the matlab code
    def transfer_style(self, input_channel, example_channel, use_mask):
        print("# Running style transfer...")
        # https://pt.wikipedia.org/wiki/Triangula%C3%A7%C3%A3o_de_Delaunay
        inputTri = scipy.spatial.Delaunay(self.input_lm)
        # exampleTri = scipy.spatial.Delaunay(self.example_lm)
        stack_depth = 6

        lStackInput, lStackExample = getLaplacianStacks(input_channel, example_channel, self.input_mask, self.example_mask, stack_depth, use_mask)
        input_residual = getResidual(input_channel, self.input_mask, stack_depth)
        example_residual = getResidual(example_channel, self.example_mask, stack_depth)


        inputEStack = getLocalEnergyStack(lStackInput)
        exampleEStack = getLocalEnergyStack(lStackExample)

        warpedStack = warpEnergyStack(exampleEStack, self.input_lm, inputTri, self.example_lm)

        gainStack = robustTransfer(lStackInput, warpedStack, inputEStack)
        warpedEResidual = warp(example_residual, self.example_lm, self.input_lm, inputTri)

        gainStack.append(warpedEResidual)
        output = sumStack(gainStack)

        return rescale(output)
    
    def extract_background(self):
        """
            Remoevs the person on the example portrair and fill in those 
            regions using image inpainting, which fill in these areas 
            seamlessly so that they blend naturally with the rest of the 
            image.
        """
        SHIFT_AMOUNT = 8
        mask = Utils.read_image(self.example_file + MASK_SUFIX, color=cv2.IMREAD_GRAYSCALE)
        print("# Extracting example background...")

        for axis in (1, 0):  # 1 for horizontal, 0 for vertical
            for shift in (SHIFT_AMOUNT, -1 * SHIFT_AMOUNT):  # Shift forward and backward
                mask = np.bitwise_or(np.roll(mask, shift, axis=axis), mask)
        
        background = cv2.inpaint(self.example, mask, 3, cv2.INPAINT_TELEA)

        return background


def getGaussianStacks(inputIm, exampleIm, stack_depth):
    gStackInput = GaussianStack(inputIm, 45, 2, stack_depth)
    gStackExample = GaussianStack(exampleIm, 45, 2, stack_depth)
    return gStackInput, gStackExample

def getLaplacianStacks(inputIm, exampleIm, input_mask, example_mask, stack_depth, use_mask):
    lStackInput = LaplacianStackAlt(inputIm, input_mask, stack_depth, use_mask)
    lStackExample = LaplacianStackAlt(exampleIm, example_mask, stack_depth, use_mask)
    return lStackInput, lStackExample

def getResidualStack(img, imgStack):
    residualStack = []
    for g in imgStack:
        residualStack.append(cv2.convolve(img, g))
    return residualStack

def getResidual(image, mask, stack_depth):
    return lowPass(image, 5*(2**stack_depth), 2**stack_depth)

def getLocalEnergyStack(lStack):
    energyStack = []
    for i in range(len(lStack)):
        laplacian = lStack[i]
        laplacian_squared = np.square(laplacian)
        energy = lowPass(laplacian_squared, 5*(2**(i+1)), 2**(i+1))
        energyStack.append(energy)
    return energyStack

def warpEnergyStack(eStack, input_shape, inputTri, example_shape):
    """
    Warp every triangle from example triangulation to input triangulation here    
    """

    warpedStack = []
    
    for elem in eStack:
        warped = warp(elem, example_shape, input_shape, inputTri)
        warpedStack.append(warped)

    return warpedStack

#Alternate approach of warping the Laplacian stacks before estimating energy
def warpLapStack(lStack, example_shape, input_shape, inputTri):
    warpedLapStack = []
    for elem in lStack:
        warped = warp(elem, example_shape, input_shape, inputTri)
        warpedLapStack.append(warped)

    return warpedLapStack

#Performs Robust transfer and gain clamping
def robustTransfer(inputLapStack, warpedStack, inputEStack):
    newGainStack = []
    e_0 = 0.01 ** 2
    gain_max = 2.8
    gain_min = 0.9
    for i in range(len(inputLapStack)):
        gain = (warpedStack[i] / (inputEStack[i] + e_0)) ** 0.5
        gain[gain > gain_max] = gain_max
        gain[gain < gain_min] = gain_min
        gain = lowPass(gain, 5*(2**i), 3*(2**i))
        newLayer = inputLapStack[i] * gain
        newGainStack.append(newLayer)

    return newGainStack

