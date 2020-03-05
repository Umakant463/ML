import numpy as np
import cv2


# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cv2.startWindowThread()

#sRead Image
og_img = cv2.imread("thermal1.jpeg")


# resizing for faster detection
og_img = cv2.resize(og_img, (640, 480))
    # using a greyscale picture, also for faster detection
gray = cv2.cvtColor(og_img, cv2.COLOR_RGB2GRAY)

# detect people in the image
    # returns the bounding boxes for the detected objects
boxes, weights = hog.detectMultiScale(og_img, winStride=(8,8) )

boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

for (xA, yA, xB, yB) in boxes:
# display the detected boxes in the colour picture
	og_img= cv2.rectangle(og_img, (xA, yA), (xB, yB),(0, 255, 0), 2)

#out.write(og_img.astype('uint8'))

cv2.imshow("HuManDetection", og_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
