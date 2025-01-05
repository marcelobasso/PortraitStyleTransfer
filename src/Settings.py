"""
Define project constants
"""

INPUTS_FOLDER = 'assets/inputs/'
OUTPUTS_FOLDER = 'assets/outputs/'
FILETYPE = '.jpg'
MASK_SUFIX = '_mask'
BACKGROUND = '_background'

### Algorithm parameters
STACKS_DEPTH = 5 # different than paper
G_MULTIPLIER = 3 # same as paper
MIN_GAIN = 0.7 # close to paper
MAX_GAIN = 3.1 # close to paper