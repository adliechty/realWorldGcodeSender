# import the necessary packages
import numpy as np
import argparse
import cv2
import time
import math
from svgpathtools import Path, Line, QuadraticBezier, CubicBezier, Arc
from svgpathtools import svg2paths, wsvg, svg2paths2, polyline
from matplotlib import pyplot as plt
from matplotlib.widgets import TextBox

global xOffset
xOffset = 0
global yOffset
yOffset = 0
global rotation
rotation = 0
global output
global origin
global xVector
global yVector
global xVectorNorm
global yVectorNorm
global matPlotImage

####################################################################################
# SHould put these in a shared libary
####################################################################################
def distanceXY(p1, p2):
  return ((p1.X - p2.X)**2 + (p1.Y - p2.Y)**2)**0.5

class Point3D:
  def __init__(self, X, Y, Z = None):
    self.X = X
    self.Y = Y
    self.Z = Z
  def to2DComplex(self):
    return self.X + self.Y * 1j
  def distanceXY(self, point):
    return distanceXY(self, point)
  def __str__(self):
    return str("(" + str(self.X) + "," + str(self.Y) + "," + str(self.Z) + ")")
  def __repr__(self):
    return str("(" + str(self.X) + "," + str(self.Y) + "," + str(self.Z) + ")")

def lineOrCurveToPoints3D(lineOrCurve, pointsPerCurve):
  if isinstance(lineOrCurve,Line):
    #print(lineOrCurve)
    return [Point3D(lineOrCurve.bpoints()[0].real, lineOrCurve.bpoints()[0].imag), \
            Point3D(lineOrCurve.bpoints()[1].real, lineOrCurve.bpoints()[1].imag)]
  elif isinstance(lineOrCurve, CubicBezier):
    points3D = []
    for i in range(int(pointsPerCurve)):
      complexPoint = lineOrCurve.point(i / (pointsPerCurve - 1.0))
      points3D.append(Point3D(complexPoint.real, complexPoint.imag, None))
    return points3D
  elif isinstance(lineOrCurve, Arc):
    points3D = []
    for i in range(int(pointsPerCurve) * 10):
      complexPoint = lineOrCurve.point(i / (pointsPerCurve * 10 - 1.0))
      points3D.append(Point3D(complexPoint.real, complexPoint.imag, None))
    return points3D

  else:
    print("unsuported type: " + str(lineOrCurve))
    quit()

def pathToPoints3D(path, pointsPerCurve):
  prevEnd = None
  points3D = []
  for lineOrCurve in path:
    curPoints3D = lineOrCurveToPoints3D(lineOrCurve, pointsPerCurve)
    #check that the last line ending starts the beginning of the next line.
    #print(curPoints3D)
    if prevEnd != None and distanceXY(curPoints3D[0], prevEnd) > 0.01:
      print(curPoints3D[0])
      print(prevEnd)
      print("A SVG path must be contiguous, one line ending and beginning on the next line.  Make a seperate path out of non contiguous lines")
      quit()
    elif prevEnd == None:
      #first line, store both beginning point and end point
      points3D.extend(curPoints3D)
    else:
      #add to point list except first one as it was verified to be same as ending of last
      points3D.extend(curPoints3D[1:])
    prevEnd = curPoints3D[-1]
  return points3D
####################################################################################

def dist(p1, p2):
  return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def scalePoints(points, scaleX, scaleY):
  for point in points:
    point.X = point.X * scaleX
    point.Y = point.Y * scaleY

def offsetPoints(points, X, Y):
  for point in points:
    point.X = point.X + X
    point.Y = point.Y + Y

def overlaySvg(image, origin, xVector, yVector, xOff = 0, yOff = 0, xOffPixel = 0, yOffPixel = 0):
  global xVectorNorm
  global yVectorNorm
  overlay = image.copy()
  # 889mm between the dots, calculate number of pixels per mm
  xLineEnd = origin + xVector
  yLineEnd = origin + yVector
  cv2.line(overlay, origin.astype(np.int), xLineEnd.astype(np.int), (0, 0, 255), 3)
  cv2.line(overlay, origin.astype(np.int), yLineEnd.astype(np.int), (0, 255, 0), 3)
  #33mm Y
  #40mm X
  xPixelPerMm = dist((0, 0), xVector) / (40 * 25.4)
  yPixelPerMm = dist((0, 0), yVector) / (33 * 25.4)
  pixelsPerInch = (xPixelPerMm + yPixelPerMm) / 2.0 * 25.4
  print(xPixelPerMm)
  print(yPixelPerMm)

  xVectorNorm = [x / dist((0, 0), xVector) for x in xVector]
  yVectorNorm = [y / dist((0, 0), yVector) for y in yVector]

  paths, attributes, svg_attributes = svg2paths2("C:\\Git\\svgToGCode\\project_StorageBox\\0p5in_BoxBacks_x4_35by32.svg")
  #paths, attributes, svg_attributes = svg2paths2("test.svg")
  
  for path in paths:
    points = pathToPoints3D(path, 10)
    #offset is in inches, convert to mm, which is what svg is in
    offsetPoints(points, xOff * 25.4, yOff * 25.4)
    scalePoints(points, xPixelPerMm, yPixelPerMm)
    offsetPoints(points, xOffPixel, yOffPixel)
    prevPoint = None
    for point in points:
      newPoint = origin + \
                 np.matmul([[xVectorNorm[0], yVectorNorm[0]], \
                            [xVectorNorm[1], yVectorNorm[1]]], \
                            [point.X, point.Y])
      #print(newPoint)
      if prevPoint is not None:
        cv2.line(overlay, prevPoint.astype(np.int), newPoint.astype(np.int), (255, 0, 0), int(pixelsPerInch * 0.25))
      prevPoint = newPoint
  return overlay
      

def updateXOffset(text):
  global xOffset
  global yOffset
  xOffset = float(text)
  overlay = overlaySvg(output, origin, xVector, yVector, xOffset, yOffset)
  matPlotImage.set_data(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
  print("inX: " + str(xOffset))
  print("Y: " + str(yOffset))
  
def updateYOffset(text):
  global xOffset
  global yOffset
  yOffset = float(text)
  overlay = overlaySvg(output, origin, xVector, yVector, xOffset, yOffset)
  matPlotImage.set_data(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
  print("X: " + str(xOffset))
  print("inY: " + str(yOffset))

def updateRotation(text):
  rotation = float(rotation)

def onclick(event):
  global origin
  global xVectorNorm
  global yVectorNorm
  print(str(event.xdata) + " " + str(event.ydata))
  print("     " + str(origin))
  #X is mirrored for matplotlib, so do origin - x for X.
  pixelsToOrigin = np.array([[event.xdata - origin[0]], [event.ydata - origin[1]]])
  newPointMm = np.matmul(np.linalg.inv([[xVectorNorm[0], yVectorNorm[0]], \
                                        [xVectorNorm[1], yVectorNorm[1]]]), pixelsToOrigin)
  newPointIn = newPointMm / 25.4
  print("  newPointIn " + str(newPointIn))
  print("      " + str(newPointIn[0][0]))
  print("      " + str(newPointIn[1][0]))
  

  overlay = overlaySvg(output, origin, xVector, yVector, newPointIn[0][0], newPointIn[1][0])
  matPlotImage.set_data(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

#############################################################################
# Main
#############################################################################

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) # Set Capture Device, in case of a USB Webcam try 1, or give -1 to get a list of available devices
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

#Set Width and Height 
# cap.set(3,1280)
# cap.set(4,720)

# The above step is to set the Resolution of the Video. The default is 640x480.
# This example works with a Resolution of 640x480.

#Wait until 3 circles are found (contour for circle, contour for mask around circle)
contours = []
while len(contours) != 6:
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
origin = np.array(centers[0])
xVector = np.array([centers[1][0] - centers[0][0], centers[1][1] - centers[0][1]])
yVector = np.array([centers[2][0] - centers[0][0], centers[2][1] - centers[0][1]])
angle = math.atan2(xVector[0] * yVector[1] - xVector[1]*yVector[0], xVector[0]*yVector[0] + xVector[1]*yVector[1]) * 360 / 3.14159 / 2.0

# Make it so centers are ordered, origin, xAxis, yAxis
if angle >= 0:
  temp = xVector
  xVector = yVector
  yVector = temp
  
print("origin: " + str(origin))
print("xVector: " + str(xVector))
print("yVector: " + str(yVector))

overlay = overlaySvg(output, origin, xVector, yVector)


fig, ax = plt.subplots()
fig.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.axis([1280,0, 0, 800])
matPlotImage = plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
xAxes = plt.axes([0.2, 0.1, 0.2, 0.04])
xBox = TextBox(xAxes, "xOffset (in)", initial="0")
xBox.on_submit(updateXOffset)

yAxes = plt.axes([0.7, 0.1, 0.2, 0.04])
yBox = TextBox(yAxes, "yOffset (in)", initial="0")
yBox.on_submit(updateYOffset)

rAxes = plt.axes([0.2, 0.01, 0.2, 0.04])
rBox =  TextBox(rAxes, "rotation (deg)", initial="0")
rBox.on_submit(updateRotation)

cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
