import cv2
import numpy as np
from picamera2 import Picamera2

img = np.zeros((512, 512, 3), dtype=np.uint8)
cv2.imshow('Test', img)
cv2.waitKey(0)
cv2.destroyAllWindows()