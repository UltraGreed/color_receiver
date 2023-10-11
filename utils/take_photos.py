import time

import cv2

camera = cv2.VideoCapture(0)
for i in range(5):
    cv2.imwrite(f'blue{i}.png', camera.read()[1])
    time.sleep(1)