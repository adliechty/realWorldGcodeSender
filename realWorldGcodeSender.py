#TODO:  use interpolateCornersCharuco to get better accuracy on corner detection

# import the necessary packages
import numpy as np
import argparse
import cv2
import time
import math
from svgpathtools import Path, Line, QuadraticBezier, CubicBezier, Arc
from svgpathtools import svg2paths, wsvg, svg2paths2, polyline
#import matplotlib
#matplotlib.use('GTK3Agg') 
from matplotlib import pyplot as plt
from matplotlib.widgets import TextBox
from matplotlib.backend_bases import MouseButton
from copy import deepcopy
from pygcode import Machine,  GCodeRapidMove, GCodeFeedRate, GCodeLinearMove, GCodeUseMillimeters
import pygcode
from pygcode.gcodes import MODAL_GROUP_MAP
import re

import sys
#sys.path.insert(1, 'C:\\Git\\gerbil\\')
#sys.path.insert(1, 'C:\\Git\\gcode_machine\\')
sys.path.insert(1, '../svgToGCode/')
from svgToGCode import cncPathsClass
from svgToGCode import cncGcodeGeneratorClass
from svgToGCode import Point3D
from svgToGCode import signedArea

from gerbil import Gerbil
import serial.tools.list_ports

import threading
import functools


global boxWidth 
global rightBoxRef
global leftBoxRef
global bedSize
global bedViewSizePixels


boxWidth = 1.25
#These are distances from machine origin (0,0,0), right, back, upper corner.
rightBoxRef = Point3D(2.0, -35.0, 1.0)
leftBoxRef = Point3D(-37.0, -35.0, 1.0)
bedSize = Point3D(-35.0, -35.0, -3.75)
bedViewSizePixels = 1400



#First ID is upper right, which is most positive Z and most positice Y
# Z, Y
global idToLocDict
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


####################################################################################
# Should put these in a shared libary
####################################################################################
def distanceXY(p1, p2):
  return ((p1.X - p2.X)**2 + (p1.Y - p2.Y)**2)**0.5


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

def centers(x1, y1, x2, y2, r):
    q = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    x3 = (x1 + x2) / 2
    y3 = (y1 + y2) / 2

    xx = (r ** 2 - (q / 2) ** 2) ** 0.5 * (y1 - y2) / q
    yy = (r ** 2 - (q / 2) ** 2) ** 0.5 * (x2 - x1) / q
    return ((x3 + xx, y3 + yy), (x3 - xx, y3 - yy))

def define_circle(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return (None, np.inf)

    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return ((cx, cy), radius)

def arcToPoints2(startX, startY, endX, endY, midX, midY):
    points = []
    (centerX, centerY), radius = define_circle((startX, startY), (midX, midY), (endX, endY))
    startAngle = math.atan2(startY - centerY, startX - centerX)
    endAngle   = math.atan2(endY   - centerY, endX   - centerX)
    counterClockWise = 1
    if signedArea([Point3D(startX, startY), Point3D(midX, midY), Point3D(endX, endY)]) < 0:
        counterClockWise = -1
    print("center: " + str(centerX) + "," + str(centerY))
    print("start: " + str(startX) + "," + str(startY))
    print("end: " + str(endX) + "," + str(endY))
    print("angles: " + str(startAngle) + "," + str(endAngle))
    print("counterClockWise: " + str(counterClockWise))
    #for angle in np.arange(startAngle, endAngle, (clockWise * -2 + 1) * 0.1):
    angle = startAngle
    stop = False
    while stop == False:
        x = math.cos(angle) * radius + centerX
        y = math.sin(angle) * radius + centerY
        points.append(Point3D(x, y))
        prevAngle = angle
        angle = (angle +  counterClockWise * 0.1) % (2 * math.pi)
        angleDiff = ((endAngle - angle + 3 * math.pi) % (2 * math.pi)) - math.pi
        #time.sleep(0.05)
        #print(angleDiff)
        if counterClockWise == -1:
          stop = angleDiff <= 0.1 and angleDiff >=0
        else:
          stop = angleDiff >= -0.1 and angleDiff <= 0

    x = math.cos(endAngle) * radius + centerX
    y = math.sin(endAngle) * radius + centerY
    points.append(Point3D(x, y))
    return points

def arcToPoints(startX, startY, endX, endY, i, j, clockWise, curZ):
    points = []
    centerX = startX + i
    centerY = startY + j
    radius = math.dist([centerX, centerY], [startX, startY])
    startAngle = math.atan2(startY - centerY, startX - centerX)
    endAngle   = math.atan2(endY   - centerY, endX   - centerX)
    print("end: " + str(endX) + "," + str(endY))
    for angle in np.arange(startAngle, endAngle, (clockWise * -2 + 1) * 0.1):
        x = math.cos(angle) * radius + centerX
        y = math.sin(angle) * radius + centerY
        points.append(Point3D(x, y, curZ))
    x = math.cos(endAngle) * radius + centerX
    y = math.sin(endAngle) * radius + centerY
    points.append(Point3D(x, y, curZ))
    return points

def rotate(origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy

####################################################################################
# OverlayGcode class
####################################################################################
class OverlayGcode:
    def __init__(self, cv2Overhead, gCodeFile, disableSender = True):
        global bedViewSizePixels
        global bedSize
        
        self.bedViewSizePixels = bedViewSizePixels
        self.bedSize = bedSize
        self.xOffset = 0
        self.yOffset = 0
        self.rotation = 0
        self.cv2Overhead = cv2.cvtColor(cv2Overhead, cv2.COLOR_BGR2RGB)
        self.move = False
        self.previewNextDrawnPoint = False

        self.startArc = None
        self.endArc = None

        fig, ax = plt.subplots()
        fig.tight_layout()
        plt.subplots_adjust(bottom=0.01, right = 0.99)
        plt.axis([self.bedViewSizePixels,0, self.bedViewSizePixels, 0])
        plt.rcParams['keymap.back'].remove('c') # we use c for circle
        #Generate matplotlib plot from opencv image
        self.matPlotImage = plt.imshow(self.cv2Overhead)
        ###############################################
        # Generate controls for plot
        ###############################################
        xAxes = plt.axes([0.01, 0.8, 0.2, 0.04])
        self.xBox = TextBox(xAxes, "xOffset (in)", initial="0")
        label = self.xBox.ax.get_children()[1] # label is a child of the TextBox axis
        label.set_position([0.5,1]) # [x,y] - change here to set the position
        label.set_horizontalalignment('center')
        label.set_verticalalignment('bottom')
        self.xBox.on_submit(self.onUpdateXOffset)

        yAxes = plt.axes([0.01, 0.7, 0.2, 0.04])
        self.yBox = TextBox(yAxes, "yOffset (in)", initial="0")
        label = self.yBox.ax.get_children()[1] # label is a child of the TextBox axis
        label.set_position([0.5,1]) # [x,y] - change here to set the position
        label.set_horizontalalignment('center')
        label.set_verticalalignment('bottom')
        self.yBox.on_submit(self.onUpdateYOffset)

        rAxes = plt.axes([0.01, 0.6, 0.2, 0.04])
        self.rBox =  TextBox(rAxes, "rotation (deg)", initial="0")
        label = self.rBox.ax.get_children()[1] # label is a child of the TextBox axis
        label.set_position([0.5,1]) # [x,y] - change here to set the position
        label.set_horizontalalignment('center')
        label.set_verticalalignment('bottom')
        self.rBox.on_submit(self.onUpdateRotation)

        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        cid = fig.canvas.mpl_connect('button_release_event', self.onrelease)
        cid = fig.canvas.mpl_connect('motion_notify_event', self.onmousemove)
        cid = fig.canvas.mpl_connect('key_press_event', self.onkeypress)

        #Create object to handle controlling the CNC machine and sending the g codes to it
        if disableSender == False:
            self.sender = GCodeSender(gCodeFile)
        

        self.points = []
        self.drawnPoints = []
        self.laserPowers = []
        self.machine = Machine()
        
        with open(gCodeFile, 'r') as fh:
          for line_text in fh.readlines():
            line = pygcode.Line(line_text)
            prevPos = self.machine.pos
            self.machine.process_block(line.block)
            
            ######################################
            # First determine machine motion mode and power
            ######################################
            motion = str(self.machine.mode.modal_groups[MODAL_GROUP_MAP['motion']])
            sCode = str(self.machine.mode.modal_groups[MODAL_GROUP_MAP['spindle_speed']])
            power = sCode.split('S')[1]
            #Make rapid movements 0 laser power
            if motion == "G00" or motion == "G0":
              self.laserPowers.append(0.0)
            else:
              self.laserPowers.append(float(power) / 100.0)

            ######################################
            # Determine machine current unit, convert to inches
            ######################################
            unit = str(self.machine.mode.modal_groups[MODAL_GROUP_MAP['units']])
            x = None
            if unit == "G20":
              if motion == "G02" or motion == "G2" or motion == "G03" or motion == "G3":
                beforeComment = line_text.split("(")[0]
                resultX = re.search('X[+-]?([0-9]*[.])?[0-9]+', beforeComment)
                resultY = re.search('Y[+-]?([0-9]*[.])?[0-9]+', beforeComment)
                resultI = re.search('I[+-]?([0-9]*[.])?[0-9]+', beforeComment)
                resultJ = re.search('J[+-]?([0-9]*[.])?[0-9]+', beforeComment)
                if resultX != None:
                    x = float(resultX.group()[1:])
                    y = float(resultY.group()[1:])
                    i = float(resultI.group()[1:])
                    j = float(resultJ.group()[1:])
                    clockWise = "G02" in motion or "G2" in motion
              else:
                self.points.append(Point3D(self.machine.pos.X, self.machine.pos.Y, self.machine.pos.Z))
            else:
              if motion == "G02" or motion == "G2" or motion == "G03" or motion == "G3":
                resultX = re.search('X[+-]?([0-9]*[.])?[0-9]+', beforeComment)
                resultY = re.search('Y[+-]?([0-9]*[.])?[0-9]+', beforeComment)
                resultI = re.search('I[+-]?([0-9]*[.])?[0-9]+', beforeComment)
                resultJ = re.search('J[+-]?([0-9]*[.])?[0-9]+', beforeComment)
                if resultX != None:
                    x = float(resultX.group()[1:]) / 25.4
                    y = float(resultY.group()[1:]) / 25.4
                    i = float(resultI.group()[1:]) / 25.4
                    j = float(resultJ.group()[1:]) / 25.4
                    clockWise = "G02" in motion or "G2" in motion
              else:
                self.points.append(Point3D(self.machine.pos.X / 25.4, self.machine.pos.Y / 25.4, self.machine.pos.Z / 25.4))
            if x != None:
              self.points.extend(arcToPoints(prevPos.X, prevPos.Y, self.machine.pos.X, self.machine.pos.Y, i, j, clockWise, self.machine.pos.Z))
              self.laserPowers.extend([self.laserPowers[-1]] * (len(self.points) - len(self.laserPowers)))

            
        #scale mm to inches
        #self.scalePoints(self.points, 1 / 25.4, 1 / 25.4)
        #self.paths, attributes, svg_attributes = svg2paths2("C:\\Git\\svgToGCode\\project_StorageBox\\0p5in_BoxBacks_x4_35by32.svg")
        #Generate a list of all points up front, non transformed
        #self.points = []
        #for path in self.paths:
        #  newPoints = pathToPoints3D(path, 10)
        #  self.points.extend(newPoints)

        #scale mm to inches
        #self.scalePoints(self.points, 1 / 25.4, 1 / 25.4)
        self.updateOverlay()

    def set_ref_loc(self, refPoints):
        self.refPoints = refPoints
        
    def scalePoints(self, points, scaleX, scaleY):
      for point in points:
        point.X = point.X * scaleX
        point.Y = point.Y * scaleY

    def offsetPoints(self, points, X, Y):
      for point in points:
        point.X = point.X + X
        point.Y = point.Y + Y

    def rotatePoints(self, points, origin, angle):
      for point in points:
        point.X, point.Y = rotate(origin, [point.X, point.Y], angle)

    def overlaySvg(self, image, xOff = 0, yOff = 0, rotation = 0):
      """
      image is opencv image
      xOff is in inches
      yOff is in inches
      rotation is in degrees
      """
      toolWidth = round(abs(0.25 * self.bedViewSizePixels / self.bedSize.X))
      #convert to radians
      rotation = rotation * math.pi / 180
      overlay = image.copy()

      #Make copy of points before transforming them
      transformedPoints = deepcopy(self.points)
      #Apply an offset in inches
      self.offsetPoints(transformedPoints, xOff, yOff)
      #Then rotate
      self.rotatePoints(transformedPoints, [xOff, yOff], rotation)
      #Then convert to pixel location
      self.scalePoints(transformedPoints, self.bedViewSizePixels / self.bedSize.X, self.bedViewSizePixels / self.bedSize.Y)
      prevPoint = None
      for point, laserPower in zip(transformedPoints, self.laserPowers):
        newPoint = (int(point.X), int(point.Y))
        if prevPoint is not None:
          cv2.line(overlay, prevPoint, newPoint, (int(laserPower * 255), 0, 0), toolWidth)
        prevPoint = newPoint

      transformedPoints = deepcopy(self.drawnPoints)
      self.scalePoints(transformedPoints, self.bedViewSizePixels / self.bedSize.X, self.bedViewSizePixels / self.bedSize.Y)
      prevPoint = None
      for point in transformedPoints:
        newPoint = (int(point.X), int(point.Y))
        if prevPoint is not None:
          cv2.line(overlay, prevPoint, newPoint, (0, 255, 0), toolWidth)
        prevPoint = newPoint
      return overlay

    def updateOverlay(self):
        overlay = self.overlaySvg(self.cv2Overhead, self.xOffset, self.yOffset, self.rotation)
        self.matPlotImage.set_data(overlay)
        self.matPlotImage.figure.canvas.draw()

    def onUpdateXOffset(self, text):
      if self.xOffset == float(text):
        return
      self.xOffset = float(text)
      self.updateOverlay()
      
    def onUpdateYOffset(self, text):
      if self.yOffset == float(text):
        return
      self.yOffset = float(text)
      self.updateOverlay()

    def onUpdateRotation(self, text):
      if self.rotation == float(text):
        return
      self.rotation = float(text)
      self.updateOverlay()

    def onmousemove(self, event):
      self.move = True
      self.mouseX = event.xdata
      self.mouseY = event.ydata
      if self.previewNextDrawnPoint:
        # if finishing drawing an arch, then preview that instead of a line
        if self.startArc != None and self.endArc != None:
            x, y = self._get_mouse_pos_inches()
            radius = math.dist([x,y], [self.startArc.X, self.startArc.Y])
            newPoints = arcToPoints2(self.startArc.X, self.startArc.Y, self.endArc.X, self.endArc.Y, x, y)
            print(newPoints)
            #newPoints = arcToPoints(self.startArc.X, self.startArc.Y, self.endArc.X, self.endArc.Y, x - self.startArc.X, y - self.startArc.Y, False, 0)
            numNewPoints = len(newPoints)
            self.drawnPoints.extend(newPoints)
            self.updateOverlay()
            #remove added points as this is just a preview
            del self.drawnPoints[-numNewPoints:]
            

        else:
            x, y = self._get_mouse_pos_inches()
            #provision to preview first point
            firstPoint = False
            if len(self.drawnPoints) == 0:
              self.drawnPoints.append(Point3D(x, y, 0))
              firstPoint = True
            self.drawnPoints.append(Point3D(x, y, 0))
            self.updateOverlay()
            self.drawnPoints.pop()
            if firstPoint:
              self.drawnPoints.pop()


    def _get_mouse_pos_inches(self):
      x = self.mouseX / self.bedViewSizePixels * self.bedSize.X
      y = self.mouseY / self.bedViewSizePixels * self.bedSize.Y
      return x, y
    def onkeypress(self, event):
        x, y = self._get_mouse_pos_inches()
        print(event.key)
        if   event.key == 's':
            self.sender.send_file(self.xOffset, self.yOffset, self.rotation)

        elif event.key == 'h':
            self.sender.home_machine()

        elif event.key == 'z':
            #Find X, Y, and Z position of the aluminum reference block on the work piece
            #sepcify the X and Y estimated position of the reference block
            self.sender.zero_on_workpice(self.refPoints)

        elif event.key == 'm':
            self.sender.absolute_move(x, y)

        elif event.key == 'd':
            # first d turns on preview
            if not self.previewNextDrawnPoint:
                self.previewNextDrawnPoint = True
                return
            self.drawnPoints.append(Point3D(x, y, 0))
            self.updateOverlay()
            print(self.drawnPoints)
        elif event.key == 'c':
          if self.startArc == None:
              self.startArc = self.drawnPoints[-1]#Point3D(x, y)
              self.endArc = Point3D(x, y)
          else:
              print(self.startArc)
              print(self.endArc)
              newPoints = arcToPoints2(self.startArc.X, self.startArc.Y, self.endArc.X, self.endArc.Y, x, y)
              print("newPoints:" + str(newPoints))
              self.drawnPoints.extend(newPoints)
              self.startArc = None
              self.endArc = None
              self.updateOverlay()
        # erase one drawn point
        elif event.key == 'e':
            if len(self.drawnPoints) > 0:
                self.drawnPoints.pop()
            self.updateOverlay()
        # Erase all drawn points
        elif event.key == 'E':
            self.drawnPoints = []
            self.updateOverlay()
        elif event.key == 'C':
            self.sender.send_drawnPoints(self.drawnPoints)
        elif event.key == 'shift':
            self.shiftHeld = True
            print("shift")

        # if a non drawing key was pushed then exit drawing preveiw mode
        if event.key != 'd' and event.key.lower() != 'e' and event.key != 'shift' and event.key != 'c':
            if self.previewNextDrawnPoint:
                self.previewNextDrawnPoint = False
                self.updateOverlay()
        if event.key != 'c':
            self.startArc = None
            self.endArc = None


    def onclick(self, event):
      self.move = False

    def onrelease(self, event):
      global matPlotImage
      #If clicking outside region, or mouse moved since released then return
      
      if event.x < 260 or self.move == True:
        return
      pixelsToOrigin = np.array([event.xdata, event.ydata])
      if event.button == MouseButton.RIGHT:
          xIn = pixelsToOrigin[0] / self.bedViewSizePixels * self.bedSize.X
          yIn = pixelsToOrigin[1] / self.bedViewSizePixels * self.bedSize.Y
          self.rotation = math.atan2(yIn - self.yOffset, xIn - self.xOffset)
          self.rotation = self.rotation * 180 / math.pi

      else:
          self.xOffset = pixelsToOrigin[0] / self.bedViewSizePixels * self.bedSize.X
          self.yOffset = pixelsToOrigin[1] / self.bedViewSizePixels * self.bedSize.Y
      self.updateOverlay()
      self.xBox.set_val(str(self.xOffset))
      self.yBox.set_val(str(self.yOffset))
      self.rBox.set_val(str(self.rotation))

def crop_half_vertically(img):
  #cropped_img = image[,int(image.shape[1]/2):int(image.shape[1])]
  #height = img.shape[0]
  width = img.shape[1]
  # Cut the image in half
  width_cutoff = int(width // 2)
  left = img[:, :width_cutoff]
  right = img[:, width_cutoff:]
  return left, right


def sortBoxPoints(points, rightSide = True):
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
def get_id_loc(image, boxes, ids, ID):
    for box, curID in zip(boxes, ids):
        if curID != ID:
            continue
        boxPoints = box[0]
        boxPointsSorted = np.array(sortBoxPoints(boxPoints))
        return boxPointsSorted
    return None

def boxes_to_point_and_location_list(boxes, ids, image, rightSide = False):
  global boxWidth
  global idToLocDict
  pointList = []
  locations = []
  for box, ID in zip(boxes, ids):
    #IDs below 33 are on right side, skip those if looking for left side points
    if (rightSide == False and ID < 33) or \
       (rightSide == True  and ID >= 33) or \
       (ID > 65):
      continue
    boxLoc = idToLocDict[ID[0]]
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
        if rightSide:
          curLoc[0] = curLoc[0] * boxWidth + rightBoxRef.Z
          curLoc[1] = curLoc[1] * boxWidth + rightBoxRef.Y
        else:
          curLoc[0] = curLoc[0] * boxWidth + leftBoxRef.Z
          curLoc[1] = curLoc[1] * boxWidth + leftBoxRef.Y
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

def generate_dest_locations(corners, image):
  global boxWidth
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

def pixel_loc_at_cnc_bed(phyToPixel):
  global bedSize
  points = np.array([[0,0],[bedSize.Z,0],[bedSize.Z,bedSize.Y],[0,bedSize.Y]])
  return points, \
         cv2.perspectiveTransform(points.reshape(-1,1,2), phyToPixel)


def display_4_lines(pixels, frame, flip=False):
  line1 = tuple(pixels[0][0].astype(np.int))
  line2   = tuple(pixels[1][0].astype(np.int))
  if flip:
    line3   = tuple(pixels[3][0].astype(np.int))
    line4   = tuple(pixels[2][0].astype(np.int))
  else:
    line3   = tuple(pixels[2][0].astype(np.int))
    line4   = tuple(pixels[3][0].astype(np.int))
  cv2.line(frame, line1,line2,(0,255,255),3)
  cv2.line(frame, line2,line3,(0,255,255),3)
  cv2.line(frame, line3,line4,(0,255,255),3)
  cv2.line(frame, line4,line1,(0,255,255),3)

class GCodeSender:
    def __init__(self, gCodeFile):
        self.event = threading.Event()
        self.dataList = []
        self.eventList = []

        self.gerbil = Gerbil(self.gerbil_callback)
        self.gerbil.setup_logging()

        ports = serial.tools.list_ports.comports()
        for p in ports:
            print(p.device)

        self.gerbil.cnect("COM4", 115200)
        self.gerbil.poll_start()
        self.gCodeFile = gCodeFile





    def gerbil_callback(self, eventstring, *data):
        args = []
        for d in data:
            args.append(str(d))
        print("GERBIL CALLBACK: event={} data={}".format(eventstring.ljust(30), ", ".join(args)))
        self.curData = data
        self.curEvent = eventstring
        self.dataList.append(data)
        self.eventList.append(eventstring)
        print(eventstring)
        print(data)

        #indicate callback is done
        self.event.set()

    def home_machine(self):
        self.gerbil.send_immediately("$H\n")
        pass

    def set_work_coord_offset(self, x = None, y = None, z = None):
        xStr, yStr, zStr = self._get_xyz_string(x, y, z)
        self.gerbil.send_immediately("G92 " + xStr + yStr + zStr + "\n") # G38.2 only works in work coordinate systeem, so set work coordinate to 0 so we know where we are in that

    def probe(self, x = None, y = None, z = None, feed = 5.9):
        xStr, yStr, zStr = self._get_xyz_string(x, y, z)
        self.gerbil.send_immediately("G38.2 " + xStr + yStr + zStr + "F" + feed + "\n")
        M114Resp = self.waitOnGCodeComplete("PRB")

    def zero_on_workpice(self, refPoints):
        avgX = (refPoints[0][0] + refPoints[1][0] + refPoints[2][0] + refPoints[3][0]) / 4.0
        avgY = (refPoints[0][1] + refPoints[1][1] + refPoints[2][1] + refPoints[3][1]) / 4.0

        self.flushGcodeRespQue()
        self.set_inches()
        time.sleep(1)
        self.absolute_move(None, None, -0.25, feedRate = 50) # Move close to Z limit
        self.waitOnGCodeComplete("G53")
        print("Complete")
        time.sleep(1)
        return

        #Rapid traverse to above reference plate
        print("avgXY: " + str(avgX) + " " + str(avgY))
        #self.gerbil.send_immediately("G53 G1 X" + str(avgX) + " Y" + str(avgY) + "\n")
        time.sleep(1)

        #Move down medium speed to reference plate
        self.set_work_coord_offset(z = 0) # probe ony works on work coordinage system, set it to 0 so we know where we are in that
        self.probe(z = -3.75, feed = 5.9) # move down by 3.75" until probe hit
        self.set_work_coord_offset(z = 0) # Set actual 0 to probed location

        #Move up, then slowly to reference plate
        self.work_offset_move(z = 0.25, feed=100) # Move just above reference plate
        self.probe(z = 0.05, feed = 1.5)
        self.set_work_coord_offset(z = 0) # Set actual 0 to probed location
        self.work_offset_move(z = 1.0, feed=100)

        #Move up, then move to side of reference plate
        #Move down, then over to side of reference plate
        #Move back, then slowly to side of reference plate

        #Move up, then over to other side

        #Move up, then move to side of reference plate
        #Move down, then over to side of reference plate
        #Move back, then slowly to side of reference plate

        #Move up, then to center of reference plate
    def waitOnGCodeComplete(self, gCode):
      resp = None
      while resp == None:
        if len(self.dataList) == 0:
          self.event.wait()
        print("curData:" + str(self.dataList[0]))
        for data in self.dataList:
          print("    " + str(data))
          if gCode in str(data):
            resp = data
        #Remove item from list
        self.dataList.pop(0)
        self.eventList.pop(0)
        time.sleep(1)
      print("Found")
      return resp

    def flushGcodeRespQue(self):
        self.dataList = []
        self.eventList = []

    def _get_xyz_string(self, x = None, y = None, z = None):
        if x == None:
            xStr = ""
        else:
            xStr = " X" + str(X)

        if y == None:
            yStr = ""
        else:
            yStr = " Y" + str(z)

        if z == None:
            zStr = ""
        else:
            zStr = " Z" + str(z)
        return xStr, yStr, zStr

    def absolute_move(self, x = None, y = None , z = None, feedRate = 100):
        xStr, yStr, zStr = self._get_xyz_string(x, y, z)
        fStr = " F" + str(feedRate)
        self.gerbil.send_immediately("G53 G1" + xStr + yStr + zStr + fStr + "\n")

    def work_offset_move(self, x = None, y = None , z = None, feedRate = 100):
        xStr, yStr, zStr = self._get_xyz_string(x, y, z)
        fStr = " F" + str(feedRate)
        self.gerbil.send_immediately("G1" + xStr + yStr + zStr + fStr + "\n")

    def send_drawnPoints(self, points3D):
      cncPaths = cncPathsClass(points3D        = points3D,
                               pointsPerCurve  = 30,
                               distPerTab      = 8,
                               tabWidth        = 0.25,
                               cutterDiameter  = 0.25
                        )
      cncGcodeGenerator = cncGcodeGeneratorClass(cncPaths           = cncPaths,
                                           materialThickness  = 0.75,
                                           depthBelowMaterial = 0.1,
                                           depthPerPass       = 0.157,
                                           cutFeedRate        = 100,
                                           safeHeight         = 1.0,
                                           tabHeight          = 0.12
                                          )
      cncGcodeGenerator.Generate()
      self.set_inches()
      self.absolute_move(z = -0.25)
      for gCode in cncGcodeGenerator.gCodes:
          self.gerbil.stream(gCode + "\n")

      self.absolute_move(z = -0.25)

    def set_inches(self):
        self.gerbil.send_immediately("G20\n")

    def set_mm(self):
        self.gerbil.send_immediately("G21\n")

    def send_file(self, xOffset, yOffset, rotation):
        #Set to inches
        self.gerbil.send_immediately("G20\n")

        ##########################################
        #Offset work to desired offset
        ##########################################
        self.gerbil.send_immediately("G54 X" + str(xOffset) + " Y" + str(yOffset) + "\n")

        ##########################################
        #Rotate work to desired rotation
        ##########################################
        deg = -rotation * 180 / math.pi
        self.gerbil.send_immediately("G68 X0 Y0 R" + str(deg) + "\n")

        #Set back to mm, typically the units g code assumes
        self.set_mm()

        ZFound = False
        with open(self.gCodeFile, 'r') as fh:
          for line_text in fh.readlines():
            if " Z" in line_text.upper():
              ZFound = True
              break

        # if Z move found in file (not a laser cutting file), then move cutter away from workspace as first move
        # so that if it was forgotten to do that, the first rapid traverse does not run into the workpiece
        if ZFound:
          self.absolute_move(z = -0.25)

        with open(self.gCodeFile, 'r') as fh:
            for line_text in fh.readlines():
                self.gerbil.stream(line_text)

        # Turn off rotated coordinate system
        self.gerbil.send_immediately("G69\n")


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
frame = cv2.imread('cnc4.jpeg')
img = cv2.imread('cnc4.jpeg')

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

pixelLoc = [None]*2
locations = [None]*2
pixelToPhysicalLoc = [None]*2
physicalToPixelLoc = [None]*2
pixelsAtBed = [None]*2
refPointsAtBed = [None]*2

########################################
# Determine vertical homography at left (i=0) and right (i=1) side of CNC machine
########################################
for i in range(0, 2):
  pixelLoc[i],  locations[i],  frame = boxes_to_point_and_location_list(boxes, ids, frame, i == 1)
  print(ids)
  for location in locations[i]:
    print(location)

  ########################################
  #Determine forward and backward transformation through homography
  ########################################
  pixelToPhysicalLoc[i], status = cv2.findHomography(np.array(pixelLoc[i]), np.array(locations[i]))
  physicalToPixelLoc[i], status    = cv2.findHomography(np.array(locations[i]), np.array(pixelLoc[i]))

  #############################################################
  # Draw vertical box on left and right vertical region of CNC
  #############################################################
  refPointsAtBed[i], pixelsAtBed[i] = pixel_loc_at_cnc_bed(physicalToPixelLoc[i])
  display_4_lines(pixelsAtBed[i], frame)

  #out = cv2.perspectiveTransform(np.array([[32.8125*7,32.8125*1],[32.8125*8,32.8125*1],[32.8125*8,32.8125*2],[32.8125*7,32.8125*2]]).reshape(-1,1,2), backward)

##########################################################################
# Get forward and backward homography from bed location to pixel location
##########################################################################
bedPixelCorners = np.array([[float(frame.shape[1]),0.0],[float(frame.shape[1]),float(frame.shape[0])],[0.0,0.0],[0.0,float(frame.shape[0])]])
refPixels = np.array([pixelsAtBed[0][1],pixelsAtBed[0][2],pixelsAtBed[1][1],pixelsAtBed[1][2]])
bedPhysicalToPixelLoc, status    = cv2.findHomography(bedPixelCorners, refPixels)
bedPixelToPhysicalLoc, status    = cv2.findHomography(refPixels, bedPixelCorners)
  
#############################################################
# Draw box on CNC bed
#############################################################
pixels = cv2.perspectiveTransform(bedPixelCorners.reshape(-1,1,2), bedPhysicalToPixelLoc)
display_4_lines(pixels, frame, flip=True)

#############################################################
# Display bed on original image
#############################################################
gray = cv2.resize(frame, (1280, 700))
cv2.imshow('image',gray)
cv2.waitKey()
  
######################################################################
# Warp perspective to perpendicular to bed view, create overlay calss
######################################################################
gCodeFile = 'test.nc'
cv2Overhead = cv2.warpPerspective(frame, bedPixelToPhysicalLoc, (frame.shape[1], frame.shape[0]))
cv2Overhead = cv2.resize(cv2Overhead, (bedViewSizePixels, bedViewSizePixels))
GCodeOverlay = OverlayGcode(cv2Overhead, gCodeFile, True)

########################################
# Detect box location in overhead image
########################################
#Change overhead image to gray for box detection
refPixelLoc    = get_id_loc(frame, boxes, ids, 66)
refPhysicalLoc = cv2.perspectiveTransform(refPixelLoc.reshape(-1,1,2), bedPixelToPhysicalLoc)
bedPercent = refPhysicalLoc / [frame.shape[1], frame.shape[0]]
bedLoc = []
for a in bedPercent:
    bedLoc.append(a[0] * [bedSize.X, bedSize.Y])
print("Bed Loc: " + str(bedLoc))
GCodeOverlay.set_ref_loc(bedLoc)

######################################################################
# Create a G Code sender now that overlay is created
######################################################################

plt.show()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
