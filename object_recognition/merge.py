import os

import numpy as np

from image_utils import load_image_rgba, save_image_rgba


############
#  CONFIG  #
PATH_PREFIX = 'images/merge/'

PATH_PREFIX_DELETED = PATH_PREFIX + 'deleted/'
PATH_PREFIX_ORIGINAL = PATH_PREFIX + 'original/'

PATH_PREFIX_SAVE = PATH_PREFIX + 'output/'
############

counter = 0

for deleted_name in os.listdir(PATH_PREFIX_DELETED):
    if deleted_name in os.listdir(PATH_PREFIX_ORIGINAL):
        original_name = deleted_name
    elif deleted_name.replace('.png', '.jpg') in os.listdir(PATH_PREFIX_ORIGINAL):
        original_name = deleted_name.replace('.png', '.jpg')
    else:
        continue

    counter += 1
    image_deleted = load_image_rgba(PATH_PREFIX_DELETED + deleted_name)
    image_original = load_image_rgba(PATH_PREFIX_ORIGINAL + original_name)
    r_layer, g_layer, b_layer = [image_original[:, :, i] for i in range(3)]
    a_layer = image_deleted[:, :, 3]

    image_result = np.dstack((r_layer, g_layer, b_layer, a_layer))

    save_image_rgba(PATH_PREFIX_SAVE + deleted_name, image_result)

