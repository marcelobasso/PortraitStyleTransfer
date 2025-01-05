# Portrait Style Transfer #
Based off of the SIGGRAPH paper "Style Transfer for Headshot Portraits"

Written by Marcelo Basso, extending the work done by Jose Chavez and Daniel Li

Link to [paper](https://people.csail.mit.edu/yichangshih/portrait_web/)


### make_mask.py ###
This is a very simple script to make it easier to create binary masks. 

Press "p" to enter the "poligon" mode, in which you can draw the point. When done, press "q" for showing the mask and saving it. Then, you can close the window.

Arguments:
1. Entire path of image to draw mask around

Example: `python3 ./make_mask.py ./example.jpg`

### main.py ###
This file contains out algorithm for style transfer. Time of execution is approximately 4 minutes with image being of size 602x750.

Arguments:
1. Name of input image, the image to have style transfer to it. Do not include the file extension, please put this image in `images/` folder.
2. Name of example image, the image to transfer style from. Do not include the file extension, please put this image in `images/` folder.
3. Boolean value, either True or False, on whether you want result to be in grayscale.
4. Boolean value, either True or False, on whether you want to use binary mask in the Laplacian stacks

Example: `python3 ./main.py marcelo george false true`
