import os
import time

import numpy as np

from image_utils import load_image_rgba, rgba_to_hsv
from model_class import get_model_path

from config import *

#####################
# CONFIG PARAMETERS #
LOAD_PREFIX = 'images/selection_learn/'
LOAD_PREFIX_OBJ = LOAD_PREFIX + OBJ_NAME + '/'

SAVE_PREFIX = 'images/selection_learn/'
SAVE_PREFIX_OBJ = SAVE_PREFIX + OBJ_NAME + '/'
#####################


def inc_data_rgb(model, index, max_len):
    dx, dy, dz = [np.arange(-PIXEL_AREA, PIXEL_AREA + 1) for _ in range(3)]

    dx_mesh, dy_mesh, dz_mesh = np.meshgrid(dx, dy, dz)

    dist_matrix = np.square(dx_mesh) + np.square(dy_mesh) + np.square(dz_mesh)

    offset_matrix = dx_mesh * RGB_AMOUNT ** 2 + dy_mesh * RGB_AMOUNT + dz_mesh

    result_index = np.sum(np.meshgrid(index, offset_matrix[dist_matrix <= PIXEL_AREA ** 2].flatten()), axis=0)

    np.add.at(
        model,
        result_index[np.logical_and(result_index >= 0, result_index < max_len)],
        1
    )


def inc_data_hsv(model, hsv_columns):
    # Calculating distances
    dx, dy, dz = [np.arange(-PIXEL_AREA, PIXEL_AREA + 1) for _ in range(3)]

    delta_columns = np.asarray(np.meshgrid(dx, dy, dz)).reshape((3, -1)).T
    dist_columns = np.sum(np.square(delta_columns), axis=1)

    delta_columns = delta_columns[dist_columns <= PIXEL_AREA ** 2]

    # Calculating resulting HSVs with offsets
    index_matrix = np.tile(delta_columns.flatten(), (hsv_columns.shape[0], 1)) + np.tile(hsv_columns, delta_columns.shape[0])
    index_matrix = np.reshape(index_matrix, (-1, 3))

    # HSV filtering
    # Modulo of H
    index_matrix[:, 0] %= H_AMOUNT
    # Removing S and V out of borders
    index_matrix = index_matrix[np.logical_and(index_matrix[:, 1] >= 0, index_matrix[:, 1] < S_AMOUNT)]
    index_matrix = index_matrix[np.logical_and(index_matrix[:, 2] >= 0, index_matrix[:, 2] < V_AMOUNT)]

    index = index_matrix[:, 0] * S_AMOUNT * V_AMOUNT + index_matrix[:, 1] * V_AMOUNT + index_matrix[:, 2]

    np.add.at(model, index, 1)


def learn_rgb():
    data_all = np.zeros((RGB_AMOUNT ** 3))
    data_object = np.zeros((RGB_AMOUNT ** 3))
    data_non_object = np.zeros((RGB_AMOUNT ** 3))

    for image_name in os.listdir(LOAD_PREFIX_OBJ):
        if 'out' in image_name:
            continue

        time1 = time.time()

        image = load_image_rgba(LOAD_PREFIX_OBJ + image_name)

        r_layer, g_layer, b_layer = [
            np.asarray(image[:, :, i] // RGB_COMPRESSION, dtype='uint32') for i in range(3)
        ]
        a_layer = image[:, :, 3]

        index_matrix = r_layer * RGB_AMOUNT ** 2 + g_layer * RGB_AMOUNT + b_layer

        inc_data_rgb(data_all, index_matrix, data_all.shape[0])
        inc_data_rgb(data_object, index_matrix[a_layer < 128], data_all.shape[0])
        inc_data_rgb(data_non_object, index_matrix[a_layer >= 128], data_all.shape[0])
        print(f"Image: {time.time() - time1}")

    np.save(get_model_path('obj'), data_object)
    np.save(get_model_path('noobj'), data_non_object)
    np.save(get_model_path('all'), data_all)


def learn_hsv():
    data_all = np.zeros((H_AMOUNT * S_AMOUNT * V_AMOUNT))
    data_object = np.zeros((H_AMOUNT * S_AMOUNT * V_AMOUNT))
    data_non_object = np.zeros((H_AMOUNT * S_AMOUNT * V_AMOUNT))

    for image_name in os.listdir(LOAD_PREFIX_OBJ):
        if 'out' in image_name:
            continue

        time1 = time.time()

        rgba_image = load_image_rgba(LOAD_PREFIX_OBJ + image_name)
        a_layer = rgba_image[:, :, 3]

        hsv_data = np.reshape(rgba_to_hsv(rgba_image), (-1, 3))

        hsv_data[:, 0] //= H_COMPRESSION
        hsv_data[:, 1] //= S_COMPRESSION
        hsv_data[:, 2] //= V_COMPRESSION

        inc_data_hsv(data_all, hsv_data)
        inc_data_hsv(data_object, hsv_data[a_layer.flatten() < 128])
        inc_data_hsv(data_non_object, hsv_data[a_layer.flatten() >= 128])
        print(f"Image: {time.time() - time1}")

    np.save(get_model_path('obj'), data_object)
    np.save(get_model_path('noobj'), data_non_object)
    np.save(get_model_path('all'), data_all)


def main():
    time1 = time.time()
    if COLOR_SCHEME == 'RGB':
        learn_rgb()
    elif COLOR_SCHEME == 'HSV':
        learn_hsv()
    print(f"Images all: {time.time() - time1}")


if __name__ == '__main__':
    main()