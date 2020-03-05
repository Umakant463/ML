import numpy as np 
import cv2
from matplotlib import pyplot as plt


##import the image as grayscale

front_img = cv2.imread("thermal.jpeg")
front_img_gray = cv2.cvtColor(front_img, cv2.COLOR_BGR2GRAY)

## detect corners using good points to track by si thomas

corners = cv2.goodFeaturesToTrack(front_img_gray, 20, 0.2, 10)
corners = np.int0(corners)


for i in corners:
	x,y = i.ravel()
	imag = cv2.circle(front_img,(x,y),3,255,-1)


#### using haris edge detectioin algorith for it
dst_harris = cv2.cornerHarris(front_img_gray,2,3,0.04)
front_img[dst_harris>0.01*dst_harris.max()]=[0,0,255]

cv2.imshow('dst_harris',dst_harris)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.imshow(imag),plt.show()