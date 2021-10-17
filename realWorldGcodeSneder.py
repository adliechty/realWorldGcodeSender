# import the necessary packages
import numpy as np
import argparse
import cv2
import time
import math

def dist(p1, p2):
  return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

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
  
  #######################################################################
  # Get grayscale image above threshold
  #######################################################################
  mask = cv2.inRange(frame, (150, 150, 150), (255, 255, 255))
  frame = cv2.bitwise_and(frame, frame, mask=mask)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  #gray = blackWhite
  
  #######################################################################
  # filter out noise
  #######################################################################
  # apply GuassianBlur to reduce noise. medianBlur is also added for smoothening, reducing noise.
  gray = cv2.GaussianBlur(gray,(7,7),0);
  gray = cv2.medianBlur(gray,7)
  
  #######################################################################
  # Get black and white image of gray scale
  #######################################################################
  blackWhite = gray.copy()
  a,blackWhite = cv2.threshold(blackWhite, 100, 255, cv2.THRESH_BINARY)

  #######################################################################
  # Find edges
  #######################################################################
  #Adaptive Guassian Threshold is to detect sharp edges in the Image. For more information Google it.
  gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,9,10.5)
  
  kernel = np.ones((5,5),np.uint8)
  gray = cv2.erode(gray,kernel,iterations = 1)
  gray = cv2.dilate(gray,kernel,iterations = 1)

  
  #######################################################################
  # detect circles in the image
  #######################################################################
  circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50, \
                             param1=100,                       \
                             param2=15,                        \
                             minRadius=5, maxRadius=15)
  # print circles
  
  # ensure at least some circles were found
  if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
    
    # loop over the (x, y) coordinates and radius of the circles
    #######################################################################
    # Find all circles
    #######################################################################
    print()
    masks = []
    blackScores = []
    averageRgbs = []
    for (x, y, r) in circles:
      # draw the circle in the output image, then draw a rectangle in the image
      # corresponding to the center of the circle
      #print(output[y,x])

      masks.append(np.zeros((output.shape[0], output.shape[1]), np.uint8))
      cv2.circle(masks[-1], (x, y), r, (255, 255, 255), -1)
      averageRgbs.append(cv2.mean(output, mask=masks[-1])[::-1])
      blackScores.append(averageRgbs[-1][1] + averageRgbs[-1][2] + averageRgbs[-1][3])

      #print("    " + str(averageRgbs[-1]))
      #print("    " + str(blackScores[-1]))

    #######################################################################
    # Find circles that are black enough to be considered reference point
    # put larger circle around those circles to mask out entire image but that area
    #######################################################################
    referenceCircles = []
    circleMask = np.zeros((output.shape[0], output.shape[1]), np.uint8)
    for ((x, y, r), blackScore) in zip(circles, blackScores):
      #print("        " + str(blackScore))
      if blackScore < 300:
        referenceCircles.append((x,y,r))
        cv2.circle(circleMask, (x, y), 30, (255, 255, 255), -1)
    blackWhite = cv2.bitwise_and(blackWhite, blackWhite, mask=circleMask)
    contours, hierarchy = cv2.findContours(blackWhite,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #print("Contours found: " + str(len(contours)))

     
    #######################################################################
    # Find center of inside countour
    #######################################################################
    if len(contours) == 6:
      avgArea = 0
      for contour in contours:
        avgArea = avgArea + cv2.contourArea(contour)
      avgArea = avgArea / len(contours)

      insideContours = []
      centers = []
      for contour in contours:
        area = cv2.contourArea(contour)
        if area < avgArea:
          insideContours.append(contour)
          M = cv2.moments(contour)
          centers.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
          x = int(centers[-1][0])
          y = int(centers[-1][1])
          cv2.rectangle(output, (x - 1, y - 1), (x + 1, y + 1), (0, 128, 255), -1)

      cv2.drawContours(output, insideContours, -1, (0,255,0), 3)
      #print(centers)

      distToOthers = []
      distToOthers.append(dist(centers[0], centers[1]) + dist(centers[0], centers[2]))
      distToOthers.append(dist(centers[1], centers[0]) + dist(centers[1], centers[2]))
      distToOthers.append(dist(centers[2], centers[0]) + dist(centers[2], centers[1]))

      # sort centers by dist to others, such that shortest distance is first (start of vector)
      centers = [x for _, x in sorted(zip(distToOthers, centers))]
      #centers = [(100,0), (100, 100), (0,0)]
      #centers = [(100,0), (0, 0), (100,100)]
      #print(centers)
      #assume x and y vector
      origin = centers[0]
      xVector = [centers[1][0] - centers[0][0], centers[1][1] - centers[0][1]]
      yVector = [centers[2][0] - centers[0][0], centers[2][1] - centers[0][1]]
      angle = math.atan2(xVector[0] * yVector[1] - xVector[1]*yVector[0], xVector[0]*yVector[0] + xVector[1]*yVector[1]) * 360 / 3.14159 / 2.0

      # Make it so centers are ordered, origin, xAxis, yAxis
      if angle < 0:
        temp = xVector
        xVector = yVector
        yVector = temp
      
      print("origin: " + str(origin))
      print("xVector: " + str(xVector))
      print("yVector: " + str(yVector))


    #cv2.imshow('blackWhite',blackWhite)

    #######################################################################
    # Record reference circle locations if 3 reference circles found
    # Should only be 3
    #######################################################################
    #masks = []
    #if len(referenceCircles) == 3:
    #  # Find contours of black and white image
    #  im2, contours, hierarchy = cv2.findContours(blackWhite,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #  centers = []
    #  for c in contours:
    #    M = cv2.moments(c)
    #    # calculate x,y coordinate of center
    #    cX = int(M["m10"] / M["m00"])
    #    cY = int(M["m01"] / M["m00"])
    #    centers.append((cX, cY))
        

    #  centersWithinCircles = []

    #  for (x, y, r) in referenceCircles:
    #    masks.append(np.zeros((output.shape[0], output.shape[1]), np.uint8))
    #    cv2.circle(masks[-1], (x, y), r, (255, 255, 255), -1)
    #    cv2.circle(output, (x, y), r, (0, 255, 0), 1)
    #    cv2.rectangle(output, (x - 1, y - 1), (x + 1, y + 1), (0, 128, 255), -1)
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
