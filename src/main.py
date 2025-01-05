"""
This file contains out algorithm for style transfer. 
Time of execution is approximately 4 minutes with 
image being of size 602x750.

Args:
    Input image, the image to have style transfer to it. Do not include the file extension, please put this image in images/ folder.
    Example image, the image to transfer style from. Do not include the file extension, please put this image in images/ folder.
    Bool: result in grayscale.
    Bool: binary mask in the Laplacian stacks

Example: 
    $ python3 ./main.py marcelo george false true marcelo_george_test
"""

import sys
from StyleTransfer import StyleTransfer

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Error: Invalid arguments")
        print("Usage: python3 main.py <input_image> <example_image> <gray> <use_mask>")
        print("Example: python3 main.py marcelo george false true")
        sys.exit(1)
    else:
        input_img = sys.argv[1]
        example_img = sys.argv[2]
        gray_op = sys.argv[3].lower() == 'true'
        use_mask = sys.argv[4].lower() == 'true'

        st = StyleTransfer(input_img, example_img, gray_op, use_mask)
        st.run()
