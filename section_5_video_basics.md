# Section 5 Videos Basics
## Connecting to Camera
cv2 is library which incorporates wide range of functionalities including image and vide processing like changing colored image grayscale, blur image, brighten dull image, past one image on to another etc. VideoCaputre is class or blue print that is specifically designed for video capture operations, allowing to access video streams from numerous sources, such as webcamp or vide files. 
```Python
import cv2
# cap is object representing video capture device like webcamp
# that was created earlier.
cap = cv2.VideoCapture(0) # default webcamp of laptop

# usually returns floating point but converting to int
# retrive height and width of video frames using get method
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


while True:
    #tuple unpacking
    """The read() method returns two values as a 
    pair (a tuple). The first value, traditionally 
    named ret, is a boolean that indicates whether 
    the frame was successfully read. The second value, 
    traditionally named frame, is the actual image data 
    of the frame."""
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```
In order to see raw image that is color image we can comment out gray and pass frame in imshow instead of gray

Saving video file in specific location. To find the location of the filer you are in just type pwd to find file path 

```Python
import cv2
# cap is object representing video capture device like webcamp
# that was created earlier.
cap = cv2.VideoCapture(0) # default webcamp of laptop

# usually returns floating point but converting to int
# retrive height and width of video frames using get method
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter('p:\\Pi OpenCV  programming\\frame.mp4', cv2.VideoWriter_fourcc(*'DIVX'),20, (width, height))

while True:
    #tuple unpacking
    """The read() method returns two values as a 
    pair (a tuple). The first value, traditionally 
    named ret, is a boolean that indicates whether 
    the frame was successfully read. The second value, 
    traditionally named frame, is the actual image data 
    of the frame."""
    ret, frame = cap.read()

    writer.write(frame)

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
writer.release()
cv2.destroyAllWindows()
```