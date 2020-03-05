## canny edge detection


import numpy as np 
import cv2
from matplotlib import pyplot as plt 
from imutils import perspective
from imutils import contours
import imutils


## read image

og_img = cv2.imread("thermal1.jpeg")
gray_img = cv2.cvtColor(og_img,cv2.COLOR_BGR2GRAY)

gray_img = cv2.GaussianBlur(gray_img, (7, 7), 0)
## detect edge with canny edge detection
edged = cv2.Canny(gray_img, 30,150)


edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

## eliminating the background
#thresh = cv2.threshold(can_edge, 225, 255, cv2.THRESH_BINARY_INV)[1]


## use good features to track
edge2 = cv2.goodFeaturesToTrack(edged,25,0.01,5)
edge2=np.int0(edge2)

for i in edge2:
    x,y = i.ravel()
    paste= cv2.circle(edged,(x,y),3,255,-1)

#plt image and show
plt.imshow(paste), plt.show()
cv2.imshow("imshow",edged)
cv2.imshow("cntsNew",cnts)
#cv2.imshow("Thresh", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()