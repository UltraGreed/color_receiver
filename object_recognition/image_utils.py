import cv2
import numpy as np


def load_image_rgba(path):
    img = cv2.imread(path, flags=-1)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return np.asarray(img)


def save_image_rgba(path, img_array):
    img = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(path, img)


def load_image_rgb(path):
    img = cv2.imread(path, flags=-1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.asarray(img)


def save_image_rgb(path, img_array):
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def load_image_hsv(path):
    img = cv2.imread(path, flags=-1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return np.asarray(img)


def save_image_hsv(path, img_array):
    img = cv2.cvtColor(img_array, cv2.COLOR_HSV2BGR)
    cv2.imwrite(path, img)


def rgba_to_hsv(img_array):
    return cv2.cvtColor(img_array[:, :, :3], cv2.COLOR_RGB2HSV)


def rgb_to_hsv(img_array):
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)