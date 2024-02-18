#Raspberry Pi LESSON 45: Using the Raspberry Pi Camera in Bullseye with OpenCV
import cv2
from picamera2 import Picamera2

picam2 = Picamera2 # created object of picamera2 class
picam2.preview_configuration.size = (1280, 720)
picam2.preview_configuration.formate = "RG8888"
picam2.preview_configuration.align()
picam2.configure("camera")
picam2.start()

while True:
    img = picam2.capture_array()
    cv2.imshow("camera", img)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()

