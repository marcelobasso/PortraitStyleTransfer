"""
Define project constants
"""

INPUTS_FOLDER = 'assets/inputs/'
OUTPUTS_FOLDER = 'assets/outputs/'
FILETYPE = '.jpg'
MASK_SUFIX = '_mask'
BACKGROUND = '_background'

### Algorithm parameters
SHIFT_AMOUNT = 8 # used in the background extraction
STACKS_DEPTH = 5 # paper = 6
G_MULTIPLIER = 3 # paper = 3
MIN_GAIN = 0.7 # paper = 0.9
MAX_GAIN = 3.1 # paper = 2.9