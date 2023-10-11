import os
import glob
import time

from image_utils import load_image_rgb, save_image_rgb

from model_class import RGBModel, HSVModel
from config import *
from model_class import get_model_path

#####################
# CONFIG PARAMETERS #
LOAD_PREFIX = 'images/selection_test/'
LOAD_PREFIX_OBJ = LOAD_PREFIX + OBJ_NAME + '/'
LOAD_PREFIX_TRUE = LOAD_PREFIX_OBJ + 'true/'
LOAD_PREFIX_FALSE = LOAD_PREFIX_OBJ + 'false/'

SAVE_PREFIX = 'images/result_test/'
SAVE_PREFIX_TRUE_POSITIVE = SAVE_PREFIX + 'true_positive/'
SAVE_PREFIX_FALSE_POSITIVE = SAVE_PREFIX + 'false_positive/'
SAVE_PREFIX_TRUE_NEGATIVE = SAVE_PREFIX + 'true_negative/'
SAVE_PREFIX_FALSE_NEGATIVE = SAVE_PREFIX + 'false_negative/'
####################


def clear_dir(path):
    files = glob.glob(f'{path}*')
    for file in files:
        os.remove(file)


def find_obj_image(image_path, is_obj):
    time1 = time.time()
    image = load_image_rgb(image_path)

    time2 = time.time()
    is_obj_found = model.check_object(image)
    print('obj check', time.time() - time2)

    if is_obj_found and is_obj:
        print('found true positive', end=' ')
        save_prefix = SAVE_PREFIX_TRUE_POSITIVE
    elif is_obj_found and not is_obj:
        print('found false positive', end=' ')
        save_prefix = SAVE_PREFIX_FALSE_POSITIVE
    elif not is_obj_found and is_obj:
        print('found false negative', end=' ')
        save_prefix = SAVE_PREFIX_FALSE_NEGATIVE
    else:
        print('found true negative', end=' ')
        save_prefix = SAVE_PREFIX_TRUE_NEGATIVE

    print(image_file)

    # Copying original image according to our model
    original_path = save_prefix + image_file
    save_image_rgb(original_path, image)

    # Saving black and white image with detected object for debugging
    gray_file = image_file.replace('.jpg', '_gray.png')
    gray_path = save_prefix + gray_file

    image_gray = model.get_grayscale()

    save_image_rgb(gray_path, image_gray)

    print('all', time.time() - time1)


for directory in (
        SAVE_PREFIX_TRUE_NEGATIVE,
        SAVE_PREFIX_FALSE_NEGATIVE,
        SAVE_PREFIX_TRUE_POSITIVE,
        SAVE_PREFIX_FALSE_POSITIVE
):
    clear_dir(directory)

if COLOR_SCHEME == 'RGB':
    model = RGBModel(get_model_path('sub'))
elif COLOR_SCHEME == 'HSV':
    model = HSVModel(get_model_path('sub'))

if os.path.exists(LOAD_PREFIX_TRUE):
    for image_file in sorted(os.listdir(LOAD_PREFIX_TRUE)):
        find_obj_image(LOAD_PREFIX_TRUE + image_file, True)

if os.path.exists(LOAD_PREFIX_FALSE):
    for image_file in sorted(os.listdir(LOAD_PREFIX_FALSE)):
        find_obj_image(LOAD_PREFIX_FALSE + image_file, False)
