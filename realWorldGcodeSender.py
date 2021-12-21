#TODO:  use interpolateCornersCharuco to get better accuracy on corner detection

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


def overlaySvg(image, xOff = 0, yOff = 0):
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

def updateXOffset(text):
  global xOffset
  global yOffset
  global output
  xOffset = float(text)
  overlay = overlaySvg2(output, pixelToMmTransformMatrix, xOffset, yOffset)
  matPlotImage.set_data(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
  print("inX: " + str(xOffset))
  print("Y: " + str(yOffset))
  
def updateYOffset(text):
  global xOffset
  global yOffset
  global output
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
  global output

  #print(str(event.xdata) + " " + str(event.ydata))
  #print("     " + str(origin))
  #X is mirrored for matplotlib, so do origin - x for X.
  pixelsToOrigin = np.array([[event.xdata - origin[0]], [event.ydata - origin[1]]])
  #print("  newPointIn " + str(newPointIn))
  #print("      " + str(newPointIn[0][0]))
  #print("      " + str(newPointIn[1][0]))
  

  overlay = overlaySvg(output, pixelToMmTransformMatrix, pixelsToOrigin[0] / 25.4, pixelsToOrigin[1] / 25.4)
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

#First ID is upper right, which is most positive Z and most positice Y
# Z, Y
idToLocDict = {0 :[2,21],
               1 :[2,19],
               2 :[2,17],
               3 :[2,16],
               4 :[2,13],
               5 :[2,11],
               6 :[2, 9],
               7 :[2, 7],
               8 :[2, 5],
               9 :[2, 3],
               10:[2, 1],
               11:[1, 20],
               12:[1, 18],
               13:[1, 16],
               14:[1, 14],
               15:[1, 12],
               16:[1, 10],
               17:[1,  8],
               18:[1,  6],
               19:[1,  4],
               20:[1,  2],
               21:[1,  0],
               22:[0,  21],
               23:[0,  19],
               24:[0,  17],
               25:[0,  15],
               26:[0,  13],
               27:[0,  11],
               28:[0,   9],
               29:[0,   7],
               30:[0,   5],
               31:[0,   3],
               32:[0,   1],
               33:[0,  20],
               34:[0,  18],
               35:[0,  16],
               36:[0,  14],
               37:[0,  12],
               38:[0,  10],
               39:[0,   8],
               40:[0,   6],
               41:[0,   4],
               42:[0,   2],
               43:[0,   0],
               44:[1,  21],
               45:[1,  19],
               46:[1,  17],
               47:[1,  15],
               48:[1,  13],
               49:[1,  11],
               50:[1,   9],
               51:[1,   7],
               52:[1,   5],
               53:[1,   3],
               54:[1,   1],
               55:[2,  20],
               56:[2,  18],
               57:[2,  16],
               58:[2,  14],
               59:[2,  12],
               60:[2,  10],
               61:[2,   8],
               62:[2,   6],
               63:[2,   4],
               64:[2,   2],
               65:[2,   0]}

def idToLoc(ID):
  return idToLocDict[ID]

def sortBoxPoints(points, rightSide):
  #First sort by X
  sortedX = sorted(points , key=lambda k: [k[0]])
  #Then sorty by Y left and right most two X set of points
  rightTwoPoints = sorted(sortedX[2:], key=lambda k: [k[1]])
  leftTwoPoints  = sorted(sortedX[0:2], key=lambda k: [k[1]])
  if rightSide:
    minZminY = leftTwoPoints[1]
    minZmaxY = leftTwoPoints[0]
    maxZmaxY = rightTwoPoints[0]
    maxZminY = rightTwoPoints[1]
  else:
    minZminY = rightTwoPoints[1]
    minZmaxY = rightTwoPoints[0]
    maxZmaxY = leftTwoPoints[0]
    maxZminY = leftTwoPoints[1]
    
  return [minZminY, minZmaxY, maxZmaxY, maxZminY]

def boxes_to_point_and_location_list(boxes, ids, boxWidth, image, rightSide = False):
  pointList = []
  locations = []
  for box, ID in zip(boxes, ids):
    #IDs below 33 are on right side, skip those if looking for left side points
    if rightSide == False and ID < 33:
      continue
    elif rightSide == True and ID >= 33:
      continue
    boxLoc = idToLoc(ID[0])
    for boxPoints in box:
      prevX = int(boxPoints[0][0])
      prevY = int(boxPoints[0][1])
      i = 0
   
      font                   = cv2.FONT_HERSHEY_SIMPLEX
      bottomLeftCornerOfText = prevX + 100,prevY
      fontScale              = 1
      fontColor              = (0,255,255)
      thickness              = 3
      lineType               = 2

      cv2.putText(image,str(ID[0]), 
          bottomLeftCornerOfText, 
          font, 
          fontScale,
          fontColor,
          thickness,
          lineType)

      boxPointsSorted = sortBoxPoints(boxPoints, rightSide)
      for point in boxPointsSorted:
        ############################################
        # Generate list of points
        ############################################
        pointList.append(point)

        
        ############################################
        # Generate point location based on boxWidth and index within box
        ############################################
        curLoc = [0,0]
        if i == 0:
          curLoc[0] = boxLoc[0] + 0
          curLoc[1] = boxLoc[1] + 0
        elif i ==1:
          curLoc[0] = boxLoc[0] + 0
          curLoc[1] = boxLoc[1] + 1
        elif i == 2:
          curLoc[0] = boxLoc[0] + 1
          curLoc[1] = boxLoc[1] + 1
        else:
          curLoc[0] = boxLoc[0] + 1
          curLoc[1] = boxLoc[1] + 0
        curLoc[0] = curLoc[0] * boxWidth
        curLoc[1] = curLoc[1] * boxWidth
        locations.append(curLoc)

        ############################################
        # Display points on image
        ############################################
        x= int(point[0])
        y= int(point[1])
        image = cv2.arrowedLine(image, (prevX,prevY), (x,y),
                                (0,255,255), 5)

        prevX = x
        prevY = y
        i = i + 1
        
  return np.array(pointList), locations, image

def generate_dest_locations(boxWidth, corners, image):
  prevX=2000
  prevY=2000
  locations = []
  yIndex = 0
  xIndex = 0
  for corner in corners:
    x,y= corner
    x= int(x)
    y= int(y)

    #cv2.rectangle(gray, (prevX,prevY),(x,y),(i*3,0,0),-1)
    image = cv2.arrowedLine(image, (prevX,prevY), (x,y),
                                     (200,0,0), 5)
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
  return                          np.array([[0,0],[-boxWidth*8,0],[-boxWidth * 8,boxWidth*34],[0,boxWidth*34]]), \
         cv2.perspectiveTransform(np.array([[0,0],[-boxWidth*8,0],[-boxWidth * 8,boxWidth*34],[0,boxWidth*34]]).reshape(-1,1,2), backward)


#############################################################################
# Main
#############################################################################

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) # Set Capture Device, in case of a USB Webcam try 1, or give -1 to get a list of available devices
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

#Set Width and Height 
# cap.set(3,1280)
# cap.set(4,720)


# Capture frame-by-frame
#ret, frame = cap.read()
boxWidth = 32.8125
frame = cv2.imread('cnc3.jpeg')
img = cv2.imread('cnc3.jpeg')

# load the image, clone it for output, and then convert it to grayscale
      
output = frame.copy()
  
#######################################################################
# Get grayscale image above threshold
#######################################################################
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

########################################
# Get aruco box information
########################################
boxes, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100))
print("boxes")
print(boxes)
print("ids")
print(ids)

########################################
# Transform output into list of points and physical locations of those points
########################################
pixelLoc = [None]*2
locations = [None]*2
pixelToPhysicalLoc = [None]*2
physicalToPixelLoc = [None]*2
pixelsAtBed = [None]*2
refPointsAtBed = [None]*2

for i in range(0, 2):
  pixelLoc[i],  locations[i],  frame = boxes_to_point_and_location_list(boxes, ids, boxWidth, frame, i == 1)
  print(locations[i])

  ########################################
  #Determine forward and backward transformation through homography
  ########################################
  pixelToPhysicalLoc[i], status = cv2.findHomography(np.array(pixelLoc[i]), np.array(locations[i]))
  physicalToPixelLoc[i], status    = cv2.findHomography(np.array(locations[i]), np.array(pixelLoc[i]))
  #im_out = cv2.warpPerspective(gray, forward, (int(800), int(600)))
  #cv2.rectangle(im_out, (int(32.8125*7),int(32.8125*1)),(int(32.8125*8),int(32.8125*2)),(255,0,0),-1)

  #############################################################
  # Draw vertical box on left and right vertical region of CNC
  #############################################################
  refPointsAtBed[i], pixelsAtBed[i] = pixel_loc_at_cnc_bed(boxWidth, physicalToPixelLoc[i])

  #out = cv2.perspectiveTransform(np.array([[32.8125*7,32.8125*1],[32.8125*8,32.8125*1],[32.8125*8,32.8125*2],[32.8125*7,32.8125*2]]).reshape(-1,1,2), backward)
  line1 = tuple(pixelsAtBed[i][0][0].astype(np.int))
  line2   = tuple(pixelsAtBed[i][1][0].astype(np.int))
  line3   = tuple(pixelsAtBed[i][2][0].astype(np.int))
  line4   = tuple(pixelsAtBed[i][3][0].astype(np.int))
  cv2.line(frame, line1,line2,(0,255,255),3)
  cv2.line(frame, line2,line3,(0,255,255),3)
  cv2.line(frame, line3,line4,(0,255,255),3)
  cv2.line(frame, line4,line1,(0,255,255),3)

##########################################################################
# Get forward and backward homography from bed location to pixel location
##########################################################################
bedPixelCorners = np.array([[1280.0,0.0],[1280.0,700.0],[0.0,0.0],[0.0,700.0]])
refPixels = np.array([pixelsAtBed[0][1],pixelsAtBed[0][2],pixelsAtBed[1][1],pixelsAtBed[1][2]])
bedPhysicalToPixelLoc, status    = cv2.findHomography(bedPixelCorners, refPixels)
bedPixelToPhysicalLoc, status    = cv2.findHomography(refPixels, bedPixelCorners)
  
#############################################################
# Draw box on CNC bed
#############################################################
pixels = cv2.perspectiveTransform(bedPixelCorners.reshape(-1,1,2), bedPhysicalToPixelLoc)
line1 = tuple(pixels[0][0].astype(np.int))
line2   = tuple(pixels[1][0].astype(np.int))
line3   = tuple(pixels[3][0].astype(np.int))
line4   = tuple(pixels[2][0].astype(np.int))
cv2.line(frame, line1,line2,(0,255,255),3)
cv2.line(frame, line2,line3,(0,255,255),3)
cv2.line(frame, line3,line4,(0,255,255),3)
cv2.line(frame, line4,line1,(0,255,255),3)

#############################################################
# Display bed on original image
#############################################################
gray = cv2.resize(frame, (1280, 700))
cv2.imshow('image',gray)
cv2.waitKey()
  
#############################################################
# Warp perspective to perpendicular to bed view
#############################################################
im_out = cv2.warpPerspective(frame, bedPixelToPhysicalLoc, (int(1280), int(700)))
overlay = overlaySvg(im_out)

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
