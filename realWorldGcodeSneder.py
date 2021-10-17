# import the necessary packages
import numpy as np
import argparse
import cv2
import time

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) # Set Capture Device, in case of a USB Webcam try 1, or give -1 to get a list of available devices
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

#Set Width and Height 
# cap.set(3,1280)
# cap.set(4,720)

# The above step is to set the Resolution of the Video. The default is 640x480.
# This example works with a Resolution of 640x480.

while(True):
  # Capture frame-by-frame
  ret, frame = cap.read()

  # load the image, clone it for output, and then convert it to grayscale
      
  output = frame.copy()
  
  mask = cv2.inRange(frame, (150, 150, 150), (255, 255, 255))
  frame = cv2.bitwise_and(frame, frame, mask=mask)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
  # apply GuassianBlur to reduce noise. medianBlur is also added for smoothening, reducing noise.
  gray = cv2.GaussianBlur(gray,(7,7),0);
  gray = cv2.medianBlur(gray,7)
  
  #Adaptive Guassian Threshold is to detect sharp edges in the Image. For more information Google it.
  gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,9,20.5)
  
  kernel = np.ones((5,5),np.uint8)
  gray = cv2.erode(gray,kernel,iterations = 1)
  gray = cv2.dilate(gray,kernel,iterations = 1)

  #contours, hier = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
  #for cnt in contours:
  #  if 200<cv2.contourArea(cnt)<5000:
  #      cv2.drawContours(output,[cnt],0,(255,0,0),2)
  #      cv2.drawContours(mask,[cnt],0,255,-1)

  
  # detect circles in the image
  circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 550, \
                             param1=100,                       \
                             param2=15,                        \
                             minRadius=5, maxRadius=15)
  # print circles
  
  # ensure at least some circles were found
  if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
    
    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
      # draw the circle in the output image, then draw a rectangle in the image
      # corresponding to the center of the circle
      cv2.circle(output, (x, y), r, (0, 255, 0), 1)
      cv2.rectangle(output, (x - 1, y - 1), (x + 1, y + 1), (0, 128, 255), -1)
      #time.sleep(0.5)
      #print( "Column Number: ")
      #print( x)
      #print( "Row Number: ")
      #print( y)
      #print( "Radius is: ")
      #print( r)

  # Display the resulting frame
  cv2.imshow('gray',gray)
  cv2.imshow('frame',output)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
