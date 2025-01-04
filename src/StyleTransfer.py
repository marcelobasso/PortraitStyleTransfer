import numpy as np
import dlib
import cv2
import skimage as sk
import scipy.spatial
from Settings import *
import Utils as Utils

class StyleTransfer():
    def __init__(self, input_file, example_file, gray_img, use_mask):
        """
        Args:
            input_file (str): name of the input image file
            example_file (str): name of the example image file
            gray_img (bool): flag to indicate if the output should be in grayscale
            use_mask (bool): flag to indicate if the mask should be used in the Laplacian stacks
        """

        # sets options
        self.input_file = input_file
        self.example_file = example_file
        self.gray_img = gray_img
        self.use_mask = use_mask

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
            
            gray = self.transfer_style(self.input, self.example)
            gray = (sk.color.rgb2gray(background) * (1 - self.input_mask)) + (gray * self.input_mask)
            self.output = gray
        else:
            background_colors = background.transpose(2, 0, 1)
            channels = []

            for c in range(len(self.example_channels)):
                print(f"# Running for color channel {c}...")
                channels.append(self.transfer_style(self.input_channels[c], self.example_channels[c]))
                self.output.append((background_colors[c] * (1 - self.input_mask)) + (channels[c] * self.input_mask))

            self.output = np.dstack(self.output)

        cv2.imshow('result', self.output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # This is based more off of the matlab code
    def transfer_style(self, input_channel, example_channel):
        print("# --- Running style transfer ---")

        # Given a set of points in the plane, Delaunay subdivides their convex hull[1] into 
        # triangles whose circumcircles do not contain any of the points. This maximizes 
        # the size of the smallest angle in any of the triangles.
        inputTri = scipy.spatial.Delaunay(self.input_lm)
        exampleTri = scipy.spatial.Delaunay(self.example_lm)

        input_ls, example_ls = self.laplacian_stacks(input_channel, example_channel)
        input_residual, example_residual = Utils.get_residuals(input_channel, example_channel)
        input_es = self.get_local_energy_stack(input_ls)
        example_es = self.get_local_energy_stack(example_ls)
        warped_stack = Utils.warp_energy_stack(example_es, self.input_lm, inputTri, self.example_lm)
        gain_stack = self.robust_transfer(input_ls, warped_stack, input_es)
        warped_residual = Utils.warp(example_residual, self.example_lm, self.input_lm, inputTri)

        gain_stack.append(warped_residual)
        gain_stack.append(input_residual)
        output = sum(gain_stack)

        return Utils.rescale(output)
    
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
        cv2.imwrite(INPUTS_FOLDER + self.example_file + BACKGROUND + FILETYPE, background)

        return background
    
    def laplacian_stack(self, image, mask):
        """
        Generates the Lalaclian stack by applying a Gaussian over and over
        to the image, creating a Gaussian Stack, 
        """
        stack = []
        stack.append(image)

        # creates Gaussian Stack
        for i in range(1, STACKS_DEPTH):
            sigma = 2 ** i
            stack.append(Utils.lowPass(image, sigma * 5, sigma))

        # creates Laplacian Stack
        mask_mult = 1 if self.use_mask else mask
        for i in range(len(stack) - 1):
            stack[i] = Utils.rescale(stack[i] - stack[i+1] * mask_mult)

        return stack
    
    def laplacian_stacks(self, input_c, example_c):
        """
        Returns the Laplacian stacks for the input and example images.
        Args:
            input_c (np.array): input image channel
            example_c (np.array): example image channel
        """
        print("# Getting Laplacian stacks...")
        lStackInput = self.laplacian_stack(input_c, self.input_mask)
        example_ls = self.laplacian_stack(example_c, self.example_mask)
        return lStackInput, example_ls

    def get_local_energy_stack(self, laplacian_s):
        print("# Getting local energy...")
        
        stack = []
        for i in range(len(laplacian_s)):
            laplacian_squared = np.square(laplacian_s[i])
            kernel_size = 2 ** (i + 1)
            energy = Utils.lowPass(laplacian_squared, 5 * kernel_size, kernel_size)
            stack.append(energy)

        return stack
    
    def robust_transfer(self, input_ls, warped_es, input_es):
        print("# Robust transfer...")
        new_gain_stack = []
        e_0 = 0.01 ** 2
        gain_max = 2.8
        gain_min = 0.9
        
        for i in range(len(input_ls)):
            gain = (warped_es[i] / (input_es[i] + e_0)) ** 0.5
            gain[gain > gain_max] = gain_max
            gain[gain < gain_min] = gain_min
            kernel_size = 2 ** (i + 1)
            gain = Utils.lowPass(gain, 5 * kernel_size, 5 * kernel_size)
            new_layer = input_ls[i] * gain
            new_gain_stack.append(new_layer)

        return new_gain_stack
