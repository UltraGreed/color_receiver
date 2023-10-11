import numpy as np
from image_utils import save_image_rgb
from model_class import get_model_path
from config import *


COLOR_AMOUNT = tuple(RGB_AMOUNT for _ in range(3)) if COLOR_SCHEME == 'RGB' else (H_AMOUNT, S_AMOUNT, V_AMOUNT)


def get_histogram_separate(axel_ind):
    image_g = np.zeros(COLOR_AMOUNT[:axel_ind] + COLOR_AMOUNT[axel_ind + 1:], dtype='uint8')

    image_r = np.round(np.max(data_obj_norm, axis=axel_ind) * 255)
    image_b = np.round(np.max(data_non_obj_norm, axis=axel_ind) * 255)

    return np.stack((image_r.astype('uint8'), image_g, image_b.astype('uint8')), axis=-1)


def get_histogram_model(axel_ind):
    image_g = np.zeros(COLOR_AMOUNT[:axel_ind] + COLOR_AMOUNT[axel_ind + 1:], dtype='uint8')

    image_r = np.round(np.max(data_sub, axis=axel_ind) * 255)
    image_b = np.where(np.max(data_sub, axis=axel_ind) == 0, 255, 0)

    return np.stack((image_r.astype('uint8'), image_g, image_b.astype('uint8')), axis=-1)


data_obj_norm = np.load(get_model_path('obj_norm')).reshape(COLOR_AMOUNT)
data_non_obj_norm = np.load(get_model_path('noobj_norm')).reshape(COLOR_AMOUNT)
data_sub = np.load(get_model_path('sub')).reshape(COLOR_AMOUNT)

save_image_rgb('hist_sep_23.png', get_histogram_separate(0))
save_image_rgb('hist_sep_13.png', get_histogram_separate(1))
save_image_rgb('hist_sep_12.png', get_histogram_separate(2))

save_image_rgb('hist_model_23.png', get_histogram_model(0))
save_image_rgb('hist_model_13.png', get_histogram_model(1))
save_image_rgb('hist_model_12.png', get_histogram_model(2))
