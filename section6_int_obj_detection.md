# Section 6 Object detection 
![alt text](image-52.png)
![alt text](image-53.png)
![alt text](image-54.png)
![alt text](image-55.png)
![alt text](image-56.png)
![alt text](image-57.png)

## Template Matching
Simplest form of Object Detection.
Template matching is image processing technique to match small image(template) which is part of large image with the large image.

This technique is widely used for object detection projects, like product quality, vehicle tracking, robotics etc

#### Analogy
Its like specific artical or image from a newspaper page, say finding suduko from larger news paper page which involve scanning through entire newspaper page and compare each section with smaller image or specific artical/template. The goal is to find regions where the contenet of the newpaper page closely matches the desired template. 
During template matching, a correlation map is created. It's like keeping track of how well the content of the newspaper page matches your template at different locations. High values in the correlation map indicate strong matches. Once the template matching process is complete, you can identify the location where the correlation is highest. This location corresponds to the position on the newspaper page where your template (specific article or image) is located.
```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('card-2.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
```

Output:                         
![alt text](image-58.png)

Its part of the larger image **card-2**. The lareger image size is more than image size of head image. It will scann through each pixel of the image and find the match. 
```Python
head = cv2.imread('card-3.png')

head = cv2.cvtColor(head, cv2.COLOR_BGR2RGB)
plt.imshow(head)
```
Output:                      ![alt text](image-59.png)

Size difference between image and shape image
```Pytho
image.shape

head.shape
```
Output:                 
```Python
(435, 580, 3)

(80, 89, 3)
```

With method considered, heatmap is generated based on higher degree of correlation we find after scanning between main image and template image or shows where maximum values are matched.                  
The **eval** takes string say "5+8" and convert it to python form of 5 + 8 to perform python task such as arithmatic operation etc.resulting 13 as output.
```Python
my_method = eval('cv2.TM_CCOEFF')
res = cv2.matchTemplate(image, head, my_method)

plt.imshow(res)
```
Output:                             
![alt text](image-60.png)

Example or explination
```Python
# Define a mathematical expression as a string
expression = "5 + 3"

# Use eval to evaluate the string as a Python expression
result = eval(expression)

# Print the result
print("The result of the expression", expression, "is:", result)
```
Output:                             
```Python
The result of the expression 5 + 3 is: 8
```
We will find max and min values of the heatmap, max and min value location and then use that to draw red rectangle around the match of the template.

```Python 
# 6 methods for comparison in a list
methods =['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR-NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for method in methods:
     # creating copy of original image
    image_copy = image.copy()
    
    technique = eval(method)

    #template matching
    result = cv2.matchTemplate(image, head, technique)

    # Tuple unpacking
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if technique in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
     
    height, width, channels = head.shape
    
    bottom_right = (top_left[0] + width, top_left[1]+height)
    cv2.rectangle(image_copy, top_left, bottom_right, color=(255, 0, 0), thickness= 10 )

    plt.subplot(121)
    plt.imshow(result)

    plt.title('Template Matching')
    plt.subplot(122)
    plt.imshow(image_copy)
    plt.title('Detection of Template')
    plt.suptitle(method)

    plt.show()

    print('\n')
    print('\n')
```
Output:                             
![](image-61.png)
![alt text](image-62.png)

Did not find the correct match
![alt text](image-63.png)

![alt text](image-64.png)
![alt text](image-65.png)
![alt text](image-66.png)

Method 2 without uning eval function
```Python
methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

for method in methods:
    # creating a copy of the original image
    image_copy = image.copy()

    # template matching using the correct technique
    result = cv2.matchTemplate(image, head, method)

    # Tuple unpacking
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    height, width, channels = head.shape

    bottom_right = (top_left[0] + width, top_left[1] + height)
    cv2.rectangle(image_copy, top_left, bottom_right, color=(255, 0, 0), thickness=10)

    plt.subplot(121)
    plt.imshow(result)

    plt.title('Template Matching')
    plt.subplot(122)
    plt.imshow(image_copy)
    plt.title('Detection of Template')
    plt.suptitle(method)

    plt.show()

    print('\n')
    print('\n')
```