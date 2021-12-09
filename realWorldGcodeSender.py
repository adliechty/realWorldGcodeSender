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

import sys
sys.path.insert(1, '../tutorial_homography/')
#Import code from compute_homography github project
from compute_homography import compute_transorm_matrix, pixel_to_mm, mm_to_pixel


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
global xPixelPerMm
global yPixelPerMm

global pixelToMmTransformMatrix

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

def mm_to_pixel_tuple(a, b):
  return(tuple(mm_to_pixel(a, b).astype(np.int)[0:2]))


def overlaySvg3(image, xOff = 0, yOff = 0):
  overlay = image.copy()
  origin = (0,0)
  cv2.line(overlay, (0,0), (400, 0), (0,0,255), 3)

  cv2.line(overlay, (0,0), (0, 400), (0,0,255), 3)

  paths, attributes, svg_attributes = svg2paths2("C:\\Git\\svgToGCode\\project_StorageBox\\0p5in_BoxBacks_x4_35by32.svg")

  for path in paths:
    points = pathToPoints3D(path, 10)
    #offset is in inches, convert to mm, which is what svg is in
    offsetPoints(points, xOff * 25.4, yOff * 25.4)
    prevPoint = None
    for point in points:
      newPoint = (int(point.X/2), int(point.Y/2))
      if prevPoint is not None:
        cv2.line(overlay, prevPoint, newPoint, (255, 0, 0), 3)
      prevPoint = newPoint

  return overlay

def overlaySvg2(image, pixelToMmTransformMatrix, xOff = 0, yOff = 0):
  overlay = image.copy()
  print(mm_to_pixel_tuple((0,         0, 1), pixelToMmTransformMatrix))
  print(mm_to_pixel_tuple((40 * 25.4, 0, 1), pixelToMmTransformMatrix))
  origin = mm_to_pixel_tuple((0,         0, 1), pixelToMmTransformMatrix)
  corner1 = mm_to_pixel_tuple((40 * 25.4, 0, 1), pixelToMmTransformMatrix)
  print()
  print(origin)
  print()
  print(corner1)
  cv2.line(overlay, origin, corner1, (0,0,255), 3)

  cv2.line(overlay, origin, \
                    mm_to_pixel_tuple((0, 33.25 * 25.4,1), pixelToMmTransformMatrix), \
                    (0,0,255), 3)

  paths, attributes, svg_attributes = svg2paths2("C:\\Git\\svgToGCode\\project_StorageBox\\0p5in_BoxBacks_x4_35by32.svg")

  for path in paths:
    points = pathToPoints3D(path, 10)
    #offset is in inches, convert to mm, which is what svg is in
    offsetPoints(points, xOff * 25.4, yOff * 25.4)
    prevPoint = None
    for point in points:
      newPoint = mm_to_pixel_tuple((point.X, point.Y, 1), pixelToMmTransformMatrix)
      if prevPoint is not None:
        cv2.line(overlay, prevPoint, newPoint, (255, 0, 0), 3)
      prevPoint = newPoint

  return overlay

def overlaySvg(image, origin, xVector, yVector, xOff = 0, yOff = 0, xOffPixel = 0, yOffPixel = 0):
  global xVectorNorm
  global yVectorNorm
  global xPixelPerMm
  global yPixelPerMm
  overlay = image.copy()
  # 889mm between the dots, calculate number of pixels per mm
  xLineEnd = origin + xVector
  yLineEnd = origin + yVector
  cv2.line(overlay, tuple(origin.astype(np.int)), tuple(xLineEnd.astype(np.int)), (0, 0, 255), 3)
  cv2.line(overlay, tuple(origin.astype(np.int)), tuple(yLineEnd.astype(np.int)), (0, 255, 0), 3)
  #33.25mm Y
  #40mm X
  xPixelPerMm = dist((0, 0), xVector) / (40 * 25.4)
  yPixelPerMm = dist((0, 0), yVector) / (33.25 * 25.4)
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
        cv2.line(overlay, tuple(prevPoint.astype(np.int)), tuple(newPoint.astype(np.int)), (255, 0, 0), int(pixelsPerInch * 0.25))
      prevPoint = newPoint
  return overlay
      

def updateXOffset(text):
  global xOffset
  global yOffset
  xOffset = float(text)
  overlay = overlaySvg2(output, pixelToMmTransformMatrix, xOffset, yOffset)
  matPlotImage.set_data(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
  print("inX: " + str(xOffset))
  print("Y: " + str(yOffset))
  
def updateYOffset(text):
  global xOffset
  global yOffset
  yOffset = float(text)
  overlay = overlaySvg2(output, pixelToMmTransformMatrix, xOffset, yOffset)
  matPlotImage.set_data(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
  print("X: " + str(xOffset))
  print("inY: " + str(yOffset))

def updateRotation(text):
  rotation = float(rotation)

def onclick(event):
  global origin
  global xVectorNorm
  global yVectorNorm
  global xPixelPerMm  
  global yPixelPerMm

  print(str(event.xdata) + " " + str(event.ydata))
  print("     " + str(origin))
  #X is mirrored for matplotlib, so do origin - x for X.
  pixelsToOrigin = np.array([[event.xdata - origin[0]], [event.ydata - origin[1]]])
  newPointPx = np.matmul(np.linalg.inv([[xVectorNorm[0], yVectorNorm[0]], \
                                        [xVectorNorm[1], yVectorNorm[1]]]), pixelsToOrigin)
  newPointIn = np.array([newPointPx[0] / xPixelPerMm / 25.4, \
                         newPointPx[1] / yPixelPerMm / 25.4])
  print("  newPointIn " + str(newPointIn))
  print("      " + str(newPointIn[0][0]))
  print("      " + str(newPointIn[1][0]))
  

  overlay = overlaySvg2(output, pixelToMmTransformMatrix, newPointIn[0][0], newPointIn[1][0])
  matPlotImage.set_data(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

def crop_half_vertically(img):
  #cropped_img = image[,int(image.shape[1]/2):int(image.shape[1])]
  #height = img.shape[0]
  width = img.shape[1]
  # Cut the image in half
  width_cutoff = int(width // 2)
  left = img[:, :width_cutoff]
  right = img[:, width_cutoff:]
  return left, right

def generate_dest_locations(boxWidth, corners, image):
  prevX=0
  prevY=0
  locations = []
  yIndex = 0
  xIndex = 0
  for corner in corners:
    x,y= corner[0]
    x= int(x)
    y= int(y)

    #cv2.rectangle(gray, (prevX,prevY),(x,y),(i*3,0,0),-1)
    image = cv2.arrowedLine(image, (prevX,prevY), (x,y),
                                     (255,255,255), 5)
    locations.append([xIndex * boxWidth, yIndex * boxWidth])
    if xIndex == 2:
      xIndex = 0
      yIndex = yIndex + 1
    else:
      xIndex = xIndex + 1
    prevX = x
    prevY = y
  return locations, image

def pixel_loc_at_cnc_bed(boxWidth, backward):
  return cv2.perspectiveTransform(np.array([[0,0],[boxWidth*8,0],[boxWidth * 8,boxWidth*22],[0,boxWidth*22]]).reshape(-1,1,2), backward)

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
while len(contours) != 8:
  # Capture frame-by-frame
  #ret, frame = cap.read()
  frame = cv2.imread('IMG_20211208_194811847.jpg')
  img = cv2.imread('IMG_20211208_194811847.jpg')

  # load the image, clone it for output, and then convert it to grayscale
      
  output = frame.copy()
  
  #######################################################################
  # Get grayscale image above threshold
  #######################################################################
  #mask = cv2.inRange(frame, (150, 150, 150), (255, 255, 255))
  #frame = cv2.bitwise_and(frame, frame, mask=mask)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  gray1,gray2 = crop_half_vertically(gray)
  gray = gray1
  ret, corners = cv2.findChessboardCorners(gray, (3, 22), None)
  boxWidth = 32.8125

  #Generate destination locations
  locations, gray = generate_dest_locations(boxWidth, corners, gray)

  #Determine forward and backware transformation through homography
  forward, status = cv2.findHomography(np.array(corners), np.array(locations))
  backward, status = cv2.findHomography(np.array(locations), np.array(corners))
  
  im_out = cv2.warpPerspective(gray, forward, (int(800), int(1200)))
  cv2.rectangle(im_out, (int(32.8125*7),int(32.8125*1)),(int(32.8125*8),int(32.8125*2)),(255,0,0),-1)

  pixelsAtBed = pixel_loc_at_cnc_bed(boxWidth, backward)
  #out = cv2.perspectiveTransform(np.array([[32.8125*7,32.8125*1],[32.8125*8,32.8125*1],[32.8125*8,32.8125*2],[32.8125*7,32.8125*2]]).reshape(-1,1,2), backward)
  line1 = tuple(pixelsAtBed[0][0].astype(np.int))
  line2   = tuple(pixelsAtBed[1][0].astype(np.int))
  line3   = tuple(pixelsAtBed[2][0].astype(np.int))
  line4   = tuple(pixelsAtBed[3][0].astype(np.int))
  cv2.line(gray, line1,line2,(255,0,0),3)
  cv2.line(gray, line2,line3,(255,0,0),3)
  cv2.line(gray, line3,line4,(255,0,0),3)
  cv2.line(gray, line4,line1,(255,0,0),3)

  
  cv2.imshow('dst',im_out)
  gray = cv2.resize(gray, (1280, 800))
  
  cv2.imshow('image',gray)
  cv2.waitKey()
  #gray = blackWhite
  print(gray.shape)
  
  #######################################################################
  # filter out noise
  #######################################################################
  # apply GuassianBlur to reduce noise. medianBlur is also added for smoothening, reducing noise.
  #gray = cv2.GaussianBlur(gray,(7,7),0);
  #gray = cv2.medianBlur(gray,7)
  
  #######################################################################
  # Get black and white image of gray scale
  #######################################################################
  blackWhite = gray.copy()
  a,blackWhite = cv2.threshold(blackWhite, 100, 255, cv2.THRESH_BINARY)

  #######################################################################
  # Find edges
  #######################################################################
  #Adaptive Guassian Threshold is to detect sharp edges in the Image. For more information Google it.
  #gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
  #          cv2.THRESH_BINARY,9,10.5)
  
  #kernel = np.ones((5,5),np.uint8)
  #gray = cv2.erode(gray,kernel,iterations = 1)
  #gray = cv2.dilate(gray,kernel,iterations = 1)

  
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
    cv2.imshow('image',gray)
    cv2.waitKey()
    

     
#######################################################################
# Find center of inside countour
#######################################################################
avgArea = 0
for contour in contours:
  avgArea = avgArea + cv2.contourArea(contour)
avgArea = avgArea / len(contours)

insideContours = []
centers = []
i  = 0
for contour in contours:
  area = cv2.contourArea(contour)
  if area < avgArea:
    insideContours.append(contour)
    M = cv2.moments(contour)
    centers.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
    x = int(centers[-1][0])
    y = int(centers[-1][1])
    if i == 0:
      cv2.rectangle(output, (x - 1, y - 40), (x + 1, y + 1), (0, 0, 255), -1)
    elif i == 1:
      cv2.rectangle(output, (x - 1, y - 30), (x + 1, y + 1), (0, 0, 255), -1)
    elif i == 2:
      cv2.rectangle(output, (x - 1, y - 20), (x + 1, y + 1), (0, 0, 255), -1)
    else:
      cv2.rectangle(output, (x - 1, y - 10), (x + 1, y + 1), (0, 0, 255), -1)
    i = i + 1

cv2.drawContours(output, insideContours, -1, (0,255,0), 3)
#print(centers)


distToOthers = []
#distToOthers.append(dist(centers[0], centers[1]) + dist(centers[0], centers[2]))
#distToOthers.append(dist(centers[1], centers[0]) + dist(centers[1], centers[2]))
#distToOthers.append(dist(centers[2], centers[0]) + dist(centers[2], centers[1]))
#distToOthers.append(dist(centers[3], centers[2]) + dist(centers[3], centers[1]))
distToOthers = [0,3,1,4]

# sort centers by dist to others, such that shortest distance is first (start of vector)
#centers = [x for _, x in sorted(zip(distToOthers, centers))]
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

#33.25in Y
#40in    X
#pixelToMmTransformMatrix = compute_transorm_matrix( {(0,0): origin,         \
#                                                    (60 * 25.4, 0):     origin + xVector,       \
#                                                    (0, 40 * 25.4) : origin + yVector})
pixelToMmTransformMatrix = compute_transorm_matrix( {(0,0): np.array(centers[0]),         \
                                                    (40 * 25.4, 0):     np.array(centers[1]),       \
                                                    (0, 60 * 25.4) : np.array(centers[2]),
                                                    (40 * 25.4, 60 * 25.4) : np.array(centers[3])})

print("centers")
print(centers)
h, status = cv2.findHomography(np.array([[centers[0][0], centers[0][1]], \
                                                  [centers[1][0], centers[1][1]], \
                                                  [centers[2][0], centers[2][1]], \
                                                  [centers[3][0], centers[3][1]]]), \
                                                  np.array([[0,600],[400,600],[0,0],[400,0]]) \
                                                  )
im_out = cv2.warpPerspective(output, h, (int(800), int(1200)))
#print(im_out)
#x = int(25.4 * 20)
#y = int(25.4 * 20)
#cv2.rectangle(im_out, (x - 1, y - 20), (x + 1, y + 1), (0, 0, 255), -1)
#x = int(25.4 * 40)
#y = int(25.4 * 20)
#cv2.rectangle(im_out, (x - 1, y - 20), (x + 1, y + 1), (0, 0, 255), -1)
#x = int(25.4 * 40)
#y = int(25.4 * 40)
#cv2.rectangle(im_out, (x - 1, y - 20), (x + 1, y + 1), (0, 0, 255), -1)
#x = int(25.4 * 20)
#y = int(25.4 * 40)
#cv2.rectangle(im_out, (x - 1, y - 20), (x + 1, y + 1), (0, 0, 255), -1)

#cv2.imshow("warpedImage", im_out)

#overlay = overlaySvg(im_out, origin, xVector, yVector)
#overlay = overlaySvg2(imout, pixelToMmTransformMatrix)
overlay = overlaySvg3(im_out)


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
