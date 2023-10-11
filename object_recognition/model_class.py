import numpy as np

from object_recognition.config import *

from object_recognition.image_utils import rgb_to_hsv


def get_model_path(model_id, obj_name=OBJ_NAME, color_scheme=COLOR_SCHEME):
    return MODEL_DIRECTORY + f'{model_id}_{obj_name}_{color_scheme.upper()}.npy'


class ImageNotLoaded(Exception):
    pass


class InferenceModel:
    def __init__(self, get_image_weight, threshold_object_part=THRESHOLD_OBJECT, threshold_clean_part=THRESHOLD_CLEAN):
        self.get_image_weight = get_image_weight

        self._image = None

        self._threshold_object_part = threshold_object_part
        self._threshold_clean_part = threshold_clean_part

        self._threshold_object = None
        self._threshold_clean = None

        self._object_center = None
        self._object_dispersion_sq = None
        self._object_pixel_size = None

        self._image_weight_raw = None
        self._image_weight = None
        self._image_sum = None

    @property
    def image(self):
        if self._image is not None:
            return self._image
        else:
            raise ImageNotLoaded

    @image.setter
    def image(self, value):
        self._image = value

        self._threshold_object = None
        self._threshold_clean = None

        self._object_center = None
        self._object_dispersion_sq = None
        self._object_pixel_size = None

        self._image_weight_raw = None
        self._image_weight = None
        self._image_sum = None

    @property
    def threshold_object(self):
        if self._threshold_object is None:
            self._threshold_object = self._threshold_object_part * len(self.image) ** 2 * MAX_PIXEL_WEIGHT

        return self._threshold_object

    @property
    def threshold_clean(self):
        if self._threshold_clean is None:
            self._threshold_clean = self._threshold_clean_part * len(self.image) ** 2 * MAX_PIXEL_WEIGHT

        return self._threshold_clean

    @property
    def image_weight_raw(self):
        if self._image_weight_raw is None:
            self._image_weight_raw = self.get_image_weight(self.image)

        return self._image_weight_raw

    @property
    def image_weight(self):
        if self._image_weight is None:
            # Removes low weighted borders of image
            # Arrays with sums of columns and rows
            col_sums = np.sum(self.image_weight_raw, axis=0)
            row_sums = np.sum(self.image_weight_raw, axis=1)
            # Starting indexes of iterators
            up = 0
            down = self.image.shape[0] - 1
            left = 0
            right = self.image.shape[1] - 1

            up_removed = 0
            down_removed = 0
            left_removed = 0
            right_removed = 0

            while (up_removed <= self.threshold_clean / 4 or
                   down_removed <= self.threshold_clean / 4) and up != down or \
                    (left_removed <= self.threshold_clean / 4 or
                     right_removed <= self.threshold_clean / 4) and left != right:
                if up_removed <= self.threshold_clean / 4 and up != down:
                    up += 1
                    up_removed += row_sums[up]
                if down_removed <= self.threshold_clean / 4 and up != down:
                    down -= 1
                    down_removed += row_sums[down]
                if left_removed <= self.threshold_clean / 4 and left != right:
                    left += 1
                    left_removed += col_sums[left]
                if right_removed <= self.threshold_clean / 4 and left != right:
                    right -= 1
                    right_removed += col_sums[right]

            # Remaining part of image
            image_remain = self.image_weight_raw[up:down + 1, left:right + 1]

            # Pad resulting array to original size
            self._image_weight = np.pad(
                image_remain,
                ((up, self.image.shape[0] - down - 1),
                 (left, self.image.shape[1] - right - 1))
            )

        return self._image_weight

    @property
    def image_sum(self):
        if self._image_sum is None:
            self._image_sum = np.sum(self.image_weight)

        return self._image_sum

    @property
    def object_center(self):
        if self.image_sum == 0:
            return self.image.shape[0] // 2, self.image.shape[1] // 2
        if self._object_center is None:
            mean_x = np.dot(np.arange(0, self.image.shape[0]), np.sum(self.image_weight, axis=1)) / self.image_sum
            mean_y = np.dot(np.arange(0, self.image.shape[1]), np.sum(self.image_weight, axis=0)) / self.image_sum
            self._object_center = np.asarray([mean_x, mean_y])

        return self._object_center

    @property
    def object_dispersion_sq(self):
        if self._object_dispersion_sq is None:
            dispersion_x = np.sum(
                np.square(np.tile(np.arange(0, self.image.shape[0])[:, np.newaxis] - self.object_center[0], (1, self.image.shape[1]))) * self.image_weight
            ) / (self.image_sum if self.image_sum else 1)

            dispersion_y = np.sum(
                np.square(np.tile(np.arange(0, self.image.shape[1]) - self.object_center[1], (self.image.shape[0], 1))) * self.image_weight
            ) / (self.image_sum if self.image_sum else 1)

            self._object_dispersion_sq = np.asarray([dispersion_x, dispersion_y])

        return self._object_dispersion_sq

    @property
    def object_pixel_size(self):
        if self._object_pixel_size is None:
            self._object_pixel_size = np.sqrt(self.object_dispersion_sq) * PIXEL_SIZE_COEFFICIENT

        return self._object_pixel_size

    # Function returning boolean of object presence
    def check_object(self, new_image=None):
        if new_image is not None:
            self.image = new_image

        return self.image_sum >= self.threshold_object

    # Function creating a black and white array image of raw object
    def get_grayscale_raw(self):
        r_layer, g_layer = [
            np.where(np.asarray(self.image_weight_raw) == 0, 0, self.image_weight_raw * 255) for _ in range(2)
        ]
        b_layer = np.where(np.asarray(self.image_weight_raw) == 0, 255, self.image_weight_raw * 255)

        return np.dstack(tuple(np.asarray(layer, dtype='uint8') for layer in (r_layer, g_layer, b_layer)))

    # Function creating a black and white array image of object
    def get_grayscale(self):
        r_layer = self.image_weight * 255

        g_layer = np.where(
            np.logical_and(self.image_weight == 0, self.image_weight_raw != 0),
            200,
            self.image_weight * 255
        )

        b_layer = np.where(self.image_weight == 0, 255, self.image_weight * 255)

        image_grayscale = np.dstack(tuple(np.asarray(layer, dtype='uint8') for layer in (r_layer, g_layer, b_layer)))

        cross_color = np.asarray([0, 255, 0], dtype='uint8') if self.check_object() else np.asarray([255, 0, 0],
                                                                                                    dtype='uint8')

        size_x, size_y = self.object_pixel_size

        obj_x, obj_y = self.object_center
        for i in range(-int(size_x), int(size_x) + 1):
            if 0 <= int(obj_x) + i < image_grayscale.shape[0]:
                image_grayscale[int(obj_x) + i][int(obj_y)] = cross_color
        for i in range(-int(size_y), int(size_y) + 1):
            if 0 <= int(obj_y) + i < image_grayscale.shape[1]:
                image_grayscale[int(obj_x)][int(obj_y) + i] = cross_color

        return image_grayscale


class RGBModel(InferenceModel):
    def __init__(self, model_path, **kwargs):
        def get_image_weight(image):
            rgb_data = image.astype('uint32') // RGB_COMPRESSION

            index_matrix = rgb_data[:, :, 0] * RGB_AMOUNT ** 2 + rgb_data[:, :, 1] * RGB_AMOUNT + rgb_data[:, :, 2]

            return model[index_matrix]

        model = np.load(model_path)

        super().__init__(get_image_weight, **kwargs)


class HSVModel(InferenceModel):
    def __init__(self, model_path, **kwargs):
        def get_image_weight(image):
            hsv_data = rgb_to_hsv(image).astype('uint32')

            hsv_data[:, :, 0] //= H_COMPRESSION
            hsv_data[:, :, 1] //= S_COMPRESSION
            hsv_data[:, :, 2] //= V_COMPRESSION

            index_matrix = hsv_data[:, :, 0] * S_AMOUNT * V_AMOUNT + hsv_data[:, :, 1] * V_AMOUNT + hsv_data[:, :, 2]

            return model[index_matrix]

        model = np.load(model_path)

        super().__init__(get_image_weight, **kwargs)
