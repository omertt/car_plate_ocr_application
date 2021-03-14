import cv2
import numpy as np
import imutils
from findcity import findCity

img = cv2.imread(r"img_path") 
img = cv2.resize(img, (620,480))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
gray = cv2.bilateralFilter(gray, 13, 15, 15)  #for removing unwanted details
edged = cv2.Canny(gray, 30, 200) 
cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10] #sorting from big to small
screenCnt = None

for c in cnts:
    
        perimeter = cv2.arcLength(c, True) #Contour Perimeter
        approx = cv2.approxPolyDP(c, 0.018 * perimeter, True) #for approximate the shape
        
        if len(approx) == 4: #contour with 4 corner
              screenCnt = approx
              break
          
mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
new_image = cv2.bitwise_and(img,img,mask=mask)
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
#Cropped = gray[topx:bottomx+3, topy:bottomy+3]
Cropped = cv2.resize(new_image, None, fx =1, fy =1, interpolation = cv2.INTER_CUBIC)
cv2.imshow("img", Cropped)

findCity(Cropped)



