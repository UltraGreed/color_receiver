# COLORSCHEME PARAMETERS #
# Color scheme for model
COLOR_SCHEME = 'HSV'
# LOADING PARAMETERS #
MODEL_DIRECTORY = 'models/'
OBJ_NAME = "red"
# OBJECT RECOGNITION PARAMETERS #
# Part of maximum image weight sum needed to recognize object
THRESHOLD_OBJECT = 0.0005
# Part of maximum image weight sum to remove from image
THRESHOLD_CLEAN = 0.0000
# MODEL-WIDE PARAMETERS #
# Maximum possible value in model
MAX_PIXEL_WEIGHT = 1
# Dimensions of RGB color space
RGB_AMOUNT = 32
RGB_COMPRESSION = 256 // RGB_AMOUNT
# Dimensions of HSV color space
H_AMOUNT = 90
S_AMOUNT = 16
V_AMOUNT = 8
H_COMPRESSION = 180 // H_AMOUNT
S_COMPRESSION = 256 // S_AMOUNT
V_COMPRESSION = 256 // V_AMOUNT
# EDUCATION PARAMETERS #
# Area in which pixels incremented during learning
PIXEL_AREA = 2
# NORMALIZATION PARAMETERS #
# Thresholds for model normalization
UPPER_BORDER_OBJECT = 0.3
UPPER_BORDER_NON_OBJECT = 0.3
# Model value which will equal to zero chance
# Ranges from -1 to 1
LOWER_MODEL_BORDER = 0
# INFERENCE PARAMETERS
PIXEL_SIZE_COEFFICIENT = 1.3
####################
