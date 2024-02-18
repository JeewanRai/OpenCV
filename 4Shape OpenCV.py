# Raspberry Pi LESSON 47: Adding Boxes, Rectangles and Circles on Images in OpenCV
import cv2
from picamera2 import Picamera2
import time

piCam = Picamera2()

dispW = 1280
dispH = 720

piCam.preview_configuration.main.size =(dispW, dispH)
piCam.preview_configuration.main.fromat = "RGB888"
piCam.preview_configuration.transform(vflip = False)
piCam.preview_configuration.controls.FrameRate = 60
piCam.configure("preview")
piCam.start()

fps =0
pos = (30, 60) # (column, row)
font = cv2.FONT_HERSHEY_SIMPLEX 
height =1.5 
myCol =(0,0, 255) # BGR with B0, GO and R255
weight = 3 # depth of color

while True:
    tStart = time.time()
    img = piCam.capture_array()
    cv2.putText(img, str(int(fps)) + 'FPS', pos, font, height, myCol, weight) 
    cv2.imshow("Camera", img)
    if cv2.waitKey(1) == ord('q'):
        break
    tEnd = time.time()
    loopTime = tStart - tEnd
    fps = 0.9 * fps + 0.1 * (1/loopTime)
    print(int(fps))

cv2.destroyAllWindows()
