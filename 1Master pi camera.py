# Raspberry Pi LESSON 44: Getting Ready to Master the Raspberry Pi Camera
"""provide function and tools for video processing, computer vision task and ML
Some of the common functionalities include image and video capture, image processing,
object detection, feature extraction, and more. Computer vision is a field of computer
science and engineering that focuses on enabling computers to interpret, understand, and 
derive meaningful information from visual data such as images and videos. """
import cv2
print(cv2.__version__)

# setting aspect ratio, height to width ratio
dispW = 1280
dispH =720

"""Class (cv2.VideoCapture): Think of it as a blueprint for a camera.
Object (cam): Think of it as an actual camera created based on that blueprint.  
class is specifically designed for handling video capturing tasks"""
cam = cv2.VideoCapture(0) # cv2.VideoCapture means want to use VideoCapture class from OpenCV cv2 library
cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispH)

while True:
    """method reads a frame from the video source. The read method returns a tuple containing two values. 
    The first value (stat) is a boolean indicating whether the frame was successfully read (True if successful, 
    False otherwise). The second value (frame) is the actual frame that was captured. The line uses tuple 
    unpacking to assign the values from the returned tuple to the variables stat and frame. This means that 
    stat will receive the first value (success/failure status), and frame will receive the second value 
    (the captured frame)."""
    stat, frame =cam.read()
    cv2.imshow('nanoCam', frame)

# if key q is pressed and 1 means delay in milliseconds. In Python, ord is used to get the ASCII value of a character.
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

