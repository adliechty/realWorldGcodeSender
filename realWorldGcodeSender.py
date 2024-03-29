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
global rightSlope
global leftSlope
global materialThickness
global cutterDiameter

materialThickness = 0.471
cutterDiameter    = 0.125


#right side, 2.3975" from bed, 2.38" near wall.  0.326 from end of bed
#left side 3.2" from bed.  3.1375 near wall.  0.3145" from end of bed
#5.2195" / 7 = 0.745643" per box

boxWidth = 0.745642857 * 1.01
bedSize = Point3D(-35.0, -35.0, -3.75)
#These are distances from machine origin (0,0,0), right, back, upper corner.
rightBoxRef = Point3D(4.0-.17, -34.0-0.2, bedSize.Z + 2.3975 - materialThickness)
leftBoxRef = Point3D(-39.0-.17, -34.0-0.2, bedSize.Z + 3.2 - materialThickness)

#This is the height of the bottom box from the bed at the far end (near Y 0)
#as the reference squares may not be perfectly level to the bed 
rightBoxFarHeight = 2.38 - materialThickness
leftBoxFarHeight  = 3.1375 - materialThickness

#There are 20 boxes, slope is divided by 20
rightSlope = (rightBoxFarHeight - (rightBoxRef.Z - bedSize.Z)) / 20.0
leftSlope = (leftBoxFarHeight - (leftBoxRef.Z - bedSize.Z)) / 20.0

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
    def __init__(self, cv2Overhead, gCodeFile = None, svgFile = None, enableSender = True):
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
        self.mouseX = 0
        self.mouseY = 0
        self.camRefCenter = [0,0]

        self.startArc = None
        self.endArc = None

        self.refPlateMeasuredLoc = [0.0, 0.0]
        self.camRefCenter = [0.0, 0.0]

        fig, ax = plt.subplots()
        fig.tight_layout()
        plt.subplots_adjust(bottom=0.01, right = 0.99)
        plt.axis([self.bedViewSizePixels,0, self.bedViewSizePixels, 0])
        plt.rcParams['keymap.back'].remove('c') # we use c for circle
        plt.rcParams['keymap.save'].remove('s') # we use s for send
        plt.rcParams['keymap.pan'].remove('p') # we use s for send
        #Generate matplotlib plot from opencv image
        self.matPlotImage = plt.imshow(self.cv2Overhead)
        ###############################################
        # Generate controls for plot
        ###############################################
        #xAxes = plt.axes([0.01, 0.8, 0.2, 0.04])
        #self.xBox = TextBox(xAxes, "xOffset (in)", initial="0")
        #label = self.xBox.ax.get_children()[1] # label is a child of the TextBox axis
        #label.set_position([0.5,1]) # [x,y] - change here to set the position
        #label.set_horizontalalignment('center')
        #label.set_verticalalignment('bottom')
        #self.xBox.on_submit(self.onUpdateXOffset)

        #yAxes = plt.axes([0.01, 0.7, 0.2, 0.04])
        #self.yBox = TextBox(yAxes, "yOffset (in)", initial="0")
        #label = self.yBox.ax.get_children()[1] # label is a child of the TextBox axis
        #label.set_position([0.5,1]) # [x,y] - change here to set the position
        #label.set_horizontalalignment('center')
        #label.set_verticalalignment('bottom')
        #self.yBox.on_submit(self.onUpdateYOffset)

        #rAxes = plt.axes([0.01, 0.6, 0.2, 0.04])
        #self.rBox =  TextBox(rAxes, "rotation (deg)", initial="0")
        #label = self.rBox.ax.get_children()[1] # label is a child of the TextBox axis
        #label.set_position([0.5,1]) # [x,y] - change here to set the position
        #label.set_horizontalalignment('center')
        #label.set_verticalalignment('bottom')
        #self.rBox.on_submit(self.onUpdateRotation)

        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        cid = fig.canvas.mpl_connect('button_release_event', self.onrelease)
        cid = fig.canvas.mpl_connect('motion_notify_event', self.onmousemove)
        cid = fig.canvas.mpl_connect('key_press_event', self.onkeypress)

        #Create object to handle controlling the CNC machine and sending the g codes to it
        self.gCodeFile = gCodeFile
        if enableSender:
            self.sender = GCodeSender()
        

        self.points = []
        self.drawnPoints = []
        self.laserPowers = []
        self.machine = Machine()


        ############################################################################
        # If generating cut paths from an SVG File
        ############################################################################
        self.svgFile = svgFile
        if svgFile != None:
            #Generate cncPaths object based on svgFile
            #These are in mm
            self.cncPaths = cncPathsClass(inputSvgFile   = svgFile,
                                          pointsPerCurve = 30,
                                          distPerTab      = 7.87,
                                          tabWidth        = 0.25,
                                          cutterDiameter  = cutterDiameter,
                                          convertSvfToIn  = True
                                         )

            #Order cuts from inside holes to outside borders
            self.cncPaths.orderCncHolePathsFirst()
            self.cncPaths.orderPartialCutsFirst()

            self.pathIndex = -1
            self.pathOffsets = []
            for path in self.cncPaths.cncPaths:
                self.pathOffsets.append([0.0, 0.0])
                

        elif gCodeFile != None:
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
        else:
            print("Must use either svg or gCode file")
            quit()

            
        self.updateOverlay()

    def set_ref_loc(self, refPixels):
        print("refPixels: " + str(refPixels))

        refPoints = []
        for refPixel in refPixels:
            refPoints.append( self._pixel_to_inches(refPixel[0], refPixel[1]))
        self.refPoints = refPoints

        avgX = (refPoints[0][0] + refPoints[1][0] + refPoints[2][0] + refPoints[3][0]) / 4.0
        avgY = (refPoints[0][1] + refPoints[1][1] + refPoints[2][1] + refPoints[3][1]) / 4.0
        self.camRefCenter = [avgX, avgY]
        # set measured ref plate location whenever reference plat is moved
        self.refPlateMeasuredLoc = self.camRefCenter + [0]
        print("Ref Points = " + str(refPoints))
        angle = getBoxAngle(refPoints)
        print("Angle: " + str(angle * 180 / math.pi))
        
    
    def phyPointsToPixels(self, transformedPoints):
        global rightBoxRef, leftBoxRef

        # Bed drawn from Y = 0 to Y = 35, but from X at left support beam and right support beam with ref boxes on them.
        self.offsetPoints(transformedPoints, -rightBoxRef.X, 0)
        self.scalePoints(transformedPoints, \
                         self.bedViewSizePixels / (leftBoxRef.X - rightBoxRef.X), \
                         self.bedViewSizePixels / self.bedSize.Y)

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

    def overlaySvgOrGcode(self, image, xOff = 0, yOff = 0, rotation = 0):
      """
      image is opencv image
      xOff is in inches
      yOff is in inches
      rotation is in degrees
      """
      global cutterDiameter
      toolWidth = round(abs(cutterDiameter * self.bedViewSizePixels / (leftBoxRef.X - rightBoxRef.X)))
      #convert to radians
      rotation = rotation * math.pi / 180
      overlay = image.copy()

      if self.gCodeFile != None:
          #Make copy of points before transforming them
          transformedPoints = deepcopy(self.points)
          self.offsetPoints(transformedPoints, xOff, yOff)
          self.rotatePoints(transformedPoints, [xOff, yOff], rotation)
      else:
          transformedPoints = []
          self.laserPowers = []
          for cncPath, offset in zip(self.cncPaths.cncPaths, self.pathOffsets):
              newPoints = deepcopy(cncPath.points3D)

              self.offsetPoints(newPoints, offset[0] , offset[1])
              transformedPoints.extend(newPoints)
              if cncPath.color[1] == 0:
                  self.laserPowers.extend([0] + [1] * (len(cncPath.points3D) - 1))
              else:
                  self.laserPowers.extend([cncPath.color[1] / 255] * len(cncPath.points3D))
          self.rotatePoints(transformedPoints, [offset[0], offset[1]], rotation)

      #Then convert to pixel location
      self.phyPointsToPixels(transformedPoints)
      prevPoint = None
      for point, laserPower in zip(transformedPoints, self.laserPowers):
        newPoint = (int(point.X), int(point.Y))
        if prevPoint is not None:
          cv2.line(overlay, prevPoint, newPoint, (int(laserPower * 255), 0, 0), toolWidth)
        prevPoint = newPoint

      transformedPoints = deepcopy(self.drawnPoints)
      self.phyPointsToPixels(transformedPoints)
      prevPoint = None
      for point in transformedPoints:
        newPoint = (int(point.X), int(point.Y))
        if prevPoint is not None:
          cv2.line(overlay, prevPoint, newPoint, (0, 255, 0), toolWidth)
        prevPoint = newPoint
      return overlay

    def updateOverlay(self):
        overlay = self.overlaySvgOrGcode(self.cv2Overhead, self.xOffset, self.yOffset, self.rotation)
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
            x, y = self._mouse_pos_inches()
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
            x, y = self._mouse_pos_inches()
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

    def _pixel_to_inches(self, x, y):
      x = x / self.bedViewSizePixels * (leftBoxRef.X - rightBoxRef.X)
      x = x + rightBoxRef.X
      y = y / self.bedViewSizePixels * self.bedSize.Y
      # Make reference plate measured center location in image be exact measured location
      x = x - self.camRefCenter[0] + self.refPlateMeasuredLoc[0]
      y = y - self.camRefCenter[1] + self.refPlateMeasuredLoc[1]
      return x, y
    def _mouse_pos_inches(self):
      print(self.mouseX)
      print(self.bedViewSizePixels)
      return self._pixel_to_inches(self.mouseX, self.mouseY)

    def onkeypress(self, event):
        x, y = self._mouse_pos_inches()
        print(event.key)
        # if sending a g code file
        if   event.key == 'g':
            self.sender.send_file(self.gCodeFile, self.xOffset, self.yOffset, self.rotation)
        # if sending an svf file
        elif event.key == 's':
            # perform transfomrations that were done in GUI on actual points in the path
            rotation = self.rotation * math.pi / 180
            for path, offset in zip(self.cncPaths.cncPaths, self.pathOffsets):
              print("before: " + str(path.points3D[0]) + " offset: " + str(offset))
              self.offsetPoints(path.points3D, offset[0] , offset[1])
              self.rotatePoints(path.points3D, [self.pathOffsets[-1][0], self.pathOffsets[-1][1]], rotation)
              print("after: " + str(path.points3D[0]))

            # Before sending add tabs
            #self.cncPaths.addTabs()
            #self.cncPaths.ModifyPointsFromTabLocations()
                
            self.sender.send_svf(self.cncPaths)
        elif event.key == 'n':
            self.pathIndex = self.pathIndex + 1
        elif event.key == 'p':
            self.pathIndex = self.pathIndex - 1

        elif event.key == 'h':
            self.sender.home_machine()

        elif event.key == 'z':
            #Find X, Y, and Z position of the aluminum reference block on the work piece
            #sepcify the X and Y estimated position of the reference block
            #self.refPlateMeasuredLoc = self.sender.zero_on_refPlate(self.refPoints)
            #Just probe z height for now for demo
            self.sender.zero_on_refPlate(self.refPoints, True)
            print("refPlateMeasuredLoc: " + str(self.refPlateMeasuredLoc))
            print("camRefCenter: " + str(self.camRefCenter))

        elif event.key == 'm':
            self.sender.absolute_move(x, y, feed = 300)

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
            # offset G codes by workspace zero as G codes send relative to workspace offset
            offset = Point3D(-self.refPlateMeasuredLoc[0], \
                             -self.refPlateMeasuredLoc[1])
            self.sender.send_drawnPoints(offset, self.drawnPoints)
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
      print("event x,y: " + str(pixelsToOrigin))
      print("mouse x,y: " + str([self.mouseX, self.mouseY]))
      xIn, yIn = self._mouse_pos_inches()
      if event.button == MouseButton.RIGHT:
          #xIn = pixelsToOrigin[0] / self.bedViewSizePixels * self.bedSize.X
          #yIn = pixelsToOrigin[1] / self.bedViewSizePixels * self.bedSize.Y
          self.rotation = math.atan2(yIn - self.yOffset, xIn - self.xOffset)
          self.rotation = self.rotation - math.pi/2.0
          self.rotation = self.rotation * 180 / math.pi

      else:
          self.xOffset = xIn
          self.yOffset = yIn
          print("xin, yIn: " + str(xIn) + "," + str(yIn))
          print(str(pixelsToOrigin[0] / self.bedViewSizePixels * self.bedSize.X) + "," + \
                str(pixelsToOrigin[1] / self.bedViewSizePixels * self.bedSize.Y))
          #self.xOffset = pixelsToOrigin[0] / self.bedViewSizePixels * self.bedSize.X
          #self.yOffset = pixelsToOrigin[1] / self.bedViewSizePixels * self.bedSize.Y
          # if negative 1 then apply offset to all paths, else just selected path
          if self.pathIndex == -1:
              i = 0
              for path in self.cncPaths.cncPaths:
                  minX = 1000000000
                  minY = 1000000000
                  for point in path.points3D:
                      minX = min(minX, point.X)
                      minY = min(minY, point.Y)

                  self.pathOffsets[i] = [xIn, yIn]
                  i = i + 1
          else:
              minX = 1000000000
              minY = 1000000000
              for point in self.cncPaths.cncPaths[self.pathIndex].points3D:
                  minX = min(minX, point.X)
                  minY = min(minY, point.Y)
              self.pathOffsets[self.pathIndex]  = [xIn - minX, yIn - minY]
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

def getBoxAngle(points):
    adjacentPoints = []
    maxDistance = 0
    for point in points[1:]:
        maxDistance = max(maxDistance, math.dist(point, points[0]))
    for point in points[1:]:
        # if an adjacent point (not across from box)
        if math.dist(point, points[0]) != maxDistance:
            print("adjacen points:")
            print(points[0])
            print(point)
            angle = math.atan2(point[1] - points[0][1], point[0] - points[0][0])
            break
    print(angle * 180 / math.pi)
    # a square is square, so we will pick one of the 4 angles 0, + 90, +180, or +270
    if angle < math.pi / 4:
        angle = angle + math.pi
    if angle >= math.pi / 4:
        angle = angle - math.pi / 2
    return angle

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
          curLoc[0] = curLoc[0] * boxWidth + rightBoxRef.Z + curLoc[1] * rightSlope
          curLoc[1] = curLoc[1] * boxWidth + rightBoxRef.Y
        else:
          curLoc[0] = curLoc[0] * boxWidth + leftBoxRef.Z + curLoc[1] * leftSlope
          curLoc[1] = curLoc[1] * boxWidth + leftBoxRef.Y
        locations.append(curLoc)

        ############################################
        # Display points on image
        ############################################
        x= int(point[0])
        y= int(point[1])
        image = cv2.arrowedLine(image, (prevX,prevY), (x,y),
                                (0,255,255), 3)

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
    def __init__(self):
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
        self.set_inches()

        self.plateHeight = 0.472441 # 12mm
        self.plateWidth  = 2.75591 # 70mm
        global cutterDiameter
        self.cutterRadius   = cutterDiameter / 2.0
        self.distToKnotch = (0.984252**2 + 0.984252**2) ** 0.5 #25mm both directions to the knotch




    def gerbil_callback(self, eventstring, *data):
        args = []
        #if eventstring != 'on_vars_change' and \
        #   eventstring != 'on_progress_percent' and \
        #   eventstring != 'on_log' and \
        #   eventstring != 'on_write' and \
        #   eventstring != 'on_line_sent' and \
        #   eventstring != 'on_bufsize_change':
        #   print("GERBIL CALLBACK: " + eventstring)
        if eventstring != "on_read":
            return
        print()
        print()
        for d in data:
            args.append(str(d))
            print(d)
        print("args    event={} data={}".format(eventstring.ljust(30), ", ".join(args)))
        self.curData = data
        self.curEvent = eventstring
        self.dataList.append(data)
        self.eventList.append(eventstring)

        #indicate callback is done
        self.event.set()

    def get_absolute_pos(self):
        self.gerbil.send_immediately("?\n")
        resp = self.waitOnGCodeComplete(">")
        m = re.match("<(.*?),MPos:(.*?),WPos:(.*?)>", resp)
        mpos_parts = m.group(2).split(",")
        return (float(mpos_parts[0]), float(mpos_parts[1]), float(mpos_parts[2]))

    def home_machine(self):
        self.gerbil.send_immediately("$H\n")
        pass

    def set_work_coord_offset(self, x = None, y = None, z = None):
        xStr, yStr, zStr = self._get_xyz_string(x, y, z)
        self.gerbil.send_immediately("G54 " + xStr + yStr + zStr + "\n")
        self.gerbil.send_immediately("G54\n")

    def set_cur_pos_as(self, x = None, y = None, z = None):
        xStr, yStr, zStr = self._get_xyz_string(x, y, z)
        self.gerbil.send_immediately("G92 " + xStr + yStr + zStr + "\n")

    def probe(self, x = None, y = None, z = None, feed = 5.9):
        xStr, yStr, zStr = self._get_xyz_string(x, y, z)
        self.gerbil.send_immediately("G38.2 " + xStr + yStr + zStr + " F" + str(feed) + "\n")
        PrbResp = self.waitOnGCodeComplete("PRB")
        # Example output:  '[PRB:-0.1965,-0.1965,-2.0697:1]'
        print("PrbResp: " + str(PrbResp))
        tmp = PrbResp.split("PRB:")[1]
        tmp = tmp.split(":1")[0]
        numbers = tmp.split(",")
        print("Numbers: " + str(numbers))
        return [float(x) for x in numbers]

    def probeZSequence(self):
        plateHeight = self.plateHeight
        #Move down medium speed to reference plate
        print("***************************************4")
        print("***************************************5")
        self.probe(z = -2.75, feed = 5.9) # move down by 2.75" until probe hit
        print("***************************************6")
        self.set_cur_pos_as(z = plateHeight) # Set actual 0 to probed location
        print("***************************************7")

        #Move up, then slowly to reference plate
        self.work_offset_move(z = plateHeight + 0.1, feed=180) # Move just above reference plate
        xyz = self.probe(z = plateHeight-0.05, feed = 1.5)
        self.set_cur_pos_as(z = plateHeight) # Set actual 0 to probed location
        self.work_offset_move(z = plateHeight + 0.5, feed=180) # Move just above reference plate, clearing lip on reference plate
        # return z height of the probe
        return xyz[0], xyz[1], xyz[2] - plateHeight


    def probeXYSequence(self, plateAngle):
        cutterRadius = self.cutterRadius
        # Firxt xAxis then yAxis
        for axisAngle in [0, math.pi / 2.0]:
            angle = axisAngle + plateAngle
            
            plateHeight = self.plateHeight
            plateWidth = self.plateWidth # total width of touch plate...half of this is distance to center of touch plate
            firstSafeDist = plateWidth * 0.75
            secSafeDist = plateWidth * 0.5 + cutterRadius + 0.1 # can get a little closer second time as we already sensed edge of plate once
            probeToDist = plateWidth * 0.25

            # Move up
            self.work_offset_move(z = plateHeight + 0.5, feed=100) # Move just above reference plate, clearing lip on reference plate
            #once medium speed, once slow speed
            for feed, dist in zip([5.9, 1.5], [firstSafeDist, secSafeDist]):
                # Move to side of touch plate
                self.work_offset_move(x = math.cos(angle) * dist, y = math.sin(angle) * dist, feed=180)
                # Move below touch plate
                self.work_offset_move(z = plateHeight-0.1, feed=100)
                # Probe to the touch plate
                refPoint = self.probe(x = math.cos(angle) * probeToDist, y = math.sin(angle) * probeToDist, feed=5.9)
                # set this as new side of touch plate
                self.set_cur_pos_as(x = math.cos(angle) * (plateWidth * 0.5 + cutterRadius) , \
                                           y = math.sin(angle) * (plateWidth * 0.5 + cutterRadius))

            # Move away from plate and up, then to center of touchplate
            self.work_offset_move(x = math.cos(angle) * secSafeDist, y = math.sin(angle) * secSafeDist, feed = 100)
            self.work_offset_move(z = plateHeight + 0.5, feed=180) # Move just above reference plate, clearing lip on reference plate
        
        # we want work coord system to be center of knotch of touch plate, not center of touch plate itself.  Move there then make that zero.
        self.work_offset_move(x = math.cos(angle - math.pi/4.0) * self.distToKnotch, y = math.sin(angle - math.pi/4.0) * self.distToKnotch, feed = 400)
        self.set_cur_pos_as(x = 0, y = 0)
        return [refPoint[0] - math.cos(angle) * (plateWidth * 0.5 + cutterRadius) + math.cos(angle - math.pi/4.0) * self.distToKnotch, \
                refPoint[1] - math.sin(angle) * (plateWidth * 0.5 + cutterRadius) + math.sin(angle - math.pi/4.0) * self.distToKnotch]


    def probeAngleOfTouchPlate(self, estPlateAngle, x, y):
        plateHeight = self.plateHeight
        plateWidth = self.plateWidth # total width of touch plate...half of this is distance to center of touch plate
        firstSafeDist = plateWidth * 0.75
        secSafeDist = plateWidth * 0.625 # can get a little closer second time as we already sensed edge of plate once
        probeToDist = plateWidth * 0.25


        angle = estPlateAngle 
        # Move up
        self.work_offset_move(z = plateHeight + 0.5, feed=180) # Move just above reference plate, clearing lip on reference plate
        # Move to side of touch plate and down a quarter of the plate width

        # this routine does not adjust work offset, so need to always be conservative
        #Probe down a quarter of touch plate first
        #once medium speed, once slow speed
        distAdjust = 0
        for feed, dist in zip([5.9, 1.5], [firstSafeDist, firstSafeDist]):
            self.work_offset_move(x = math.cos(angle) * (dist - distAdjust) + math.cos(angle - math.pi/2.0) * plateWidth * 0.25 , \
                                  y = math.sin(angle) * (dist - distAdjust) + math.sin(angle - math.pi/2.0) * plateWidth * 0.25, feed=400)
            # Move below touch plate
            self.work_offset_move(z = plateHeight-0.1, feed=180)
            # Probe to the touch plate
            ref1 = self.probe(x = math.cos(angle) * probeToDist + math.cos(angle - math.pi/2.0) * plateWidth * 0.25 , \
                              y = math.sin(angle) * probeToDist + math.sin(angle - math.pi/2.0) * plateWidth * 0.25, feed=feed)
            # move just 0.1" away from plate next probe since we know where idge roughy is now
            distAdjust = dist - math.dist([x,y], [ref1[0], ref1[1]]) - 0.05
            print(ref1)
            print("distAdjust: " + str(distAdjust))

        #Probe up a quarter of touch plate second
        #once medium speed, once slow speed
        distAdjust = 0
        for feed, dist in zip([5.9, 1.5], [firstSafeDist, firstSafeDist]):
            self.work_offset_move(x = math.cos(angle) * (dist - distAdjust) + math.cos(angle + math.pi/2.0) * plateWidth * 0.25 , \
                                  y = math.sin(angle) * (dist - distAdjust) + math.sin(angle + math.pi/2.0) * plateWidth * 0.25, feed=400)
            # Move below touch plate
            self.work_offset_move(z = plateHeight - 0.1, feed=100)
            # Probe to the touch plate
            ref2 = self.probe(x = math.cos(angle) * probeToDist + math.cos(angle + math.pi/2.0) * plateWidth * 0.25 , \
                              y = math.sin(angle) * probeToDist + math.sin(angle + math.pi/2.0) * plateWidth * 0.25, feed=feed)
            distAdjust = dist - math.dist([x,y], [ref2[0], ref2[1]]) - 0.05
        
        # Move away from reference plate and up
        self.work_offset_move(x = math.cos(angle) * firstSafeDist + math.cos(angle + math.pi/2.0) * plateWidth * 0.25 , \
                              y = math.sin(angle) * firstSafeDist + math.sin(angle + math.pi/2.0) * plateWidth * 0.25, feed=400)
        self.work_offset_move(z = plateHeight + 0.5, feed=180) # Move just above reference plate, clearing lip on reference plate

        yAxisAngle = math.atan2(ref2[1] - ref1[1], ref2[0] - ref1[0])
        xAxisAngle = yAxisAngle - math.pi / 2.0 % (2 * math.pi)
        print("estPlateAngle:" + str(estPlateAngle))
        print("xAxisAngle: " + str(xAxisAngle))
        return xAxisAngle

    def probeSequence(self, estPlateAngle, justZ):
        #function assumes spindle is directly above probe plate in estimated middle
        #function returns x, y, z position of center top of plate

        # First Zero out work coord offset with best we have thus far
        self.set_cur_pos_as(x=0, y=0, z = 0) # probe ony works on work coordinage system, set it to 0 so we know where we are in that
        # first get Z height right
        x, y, z = self.probeZSequence()
        if justZ:
            #reset G92 coordinate system to normal,
            #where current position is actual position is position in work coordinate system
            self.set_cur_pos_as(x = x, y = y, z = self.plateHeight + 0.5)
            self.set_work_coord_offset(x = 0.0, y = 0.0, Z = z - self.plateHeight)
            return [x, y, z]
        plateAngle = self.probeAngleOfTouchPlate(estPlateAngle, x, y)
        xy =  self.probeXYSequence(plateAngle)
        print("xy: " + str(xy))
        self.set_work_coord_offset(x, y, z)
        return xy + [z]
            
    def zero_on_refPlate(self, refPoints, justZ = False):
        avgX = (refPoints[0][0] + refPoints[1][0] + refPoints[2][0] + refPoints[3][0]) / 4.0
        avgY = (refPoints[0][1] + refPoints[1][1] + refPoints[2][1] + refPoints[3][1]) / 4.0
        angle = getBoxAngle(refPoints)


        self.flushGcodeRespQue()
        self.set_inches()
        self.absolute_move(None, None, -0.25, feed = 180) # Move close to Z limit
        # move 1.75" away from charuco marker bottom left
        self.absolute_move(avgX + 1.335*math.cos(math.pi*5/4), avgY + 1.335*math.sin(math.pi*5/4), None,  feed = 300) # Move above estimated ref plate

        print("avgXY: " + str(avgX) + " " + str(avgY))
        #first test out zero angle, then test out actual angle
        #return self.probeSequence(0)

        return self.probeSequence(angle, justZ)

    def waitOnGCodeComplete(self, gCode):
      resp = None
      while resp == None:
        while len(self.dataList) == 0:
          self.event.wait()
        print("curData:" + str(self.dataList[0]))
        for data in self.dataList:
          print("    " + str(data))
          if gCode in str(data):
            resp = data
        #Remove item from list
        self.dataList.pop(0)
        self.eventList.pop(0)
        #time.sleep(1)
      print("resp: " + str(resp))
      print("Found: " + str(resp[0]))
      if isinstance(resp[0], dict):
          return resp[0][gCode]
      else:
          return resp[0]

    def flushGcodeRespQue(self):
        self.dataList = []
        self.eventList = []

    def _get_xyz_string(self, x = None, y = None, z = None):
        if x == None:
            xStr = ""
        else:
            xStr = " X" + str(x)

        if y == None:
            yStr = ""
        else:
            yStr = " Y" + str(y)

        if z == None:
            zStr = ""
        else:
            zStr = " Z" + str(z)
        return xStr, yStr, zStr

    def absolute_move(self, x = None, y = None , z = None, feed = 100):
        xStr, yStr, zStr = self._get_xyz_string(x, y, z)
        fStr = " F" + str(feed)
        self.set_inches()
        self.gerbil.send_immediately("G53 G1" + xStr + yStr + zStr + fStr + "\n")

    def work_offset_move(self, x = None, y = None , z = None, feed = 100):
        xStr, yStr, zStr = self._get_xyz_string(x, y, z)
        fStr = " F" + str(feed)
        self.gerbil.send_immediately("G1" + xStr + yStr + zStr + fStr + "\n")

    def send_svf(self, cncPaths):
      global materialThickness
      global cutterDiameter

      cncGcodeGenerator = cncGcodeGeneratorClass(cncPaths           = cncPaths,
                                           materialThickness  = materialThickness,
                                           depthBelowMaterial = 0.06,
                                           depthPerPass       = 0.107,
                                           cutFeedRate        = 79,
                                           safeHeight         = 0.25,
                                           tabHeight          = 0.12,
                                           useMM              = False # use inches
                                          )
      cncGcodeGenerator.Generate()
      cncGcodeGenerator.Save("test.nc")
      #asdfasdf
      self.set_inches()
      self.absolute_move(z = -0.5)
      print("SENDING GCODE")
      gCodeStrs = []
      for code in cncGcodeGenerator.gCodes:
          gCodeStrs.append(str(code))
      # put whole file in buffer then run the job
      self.gerbil.write(gCodeStrs)
      self.gerbil.job_run()
      #for gCode in cncGcodeGenerator.gCodes:
      #    code = str(gCode) + "\n"
      #    #print("CODE:" + code)
      #    self.gerbil.stream(code)
      #    #self.gerbil.send_immediately(code)
      #    time.sleep(0.01)

      self.absolute_move(z = -0.25)

    def send_drawnPoints(self, offset, points3D):
      global materialThickness
      global cutterDiameter
      points = deepcopy(points3D)
      for point in points:
          point.X = point.X + offset.X
          point.Y = point.Y + offset.Y
      cncPaths = cncPathsClass(points3D        = points,
                               pointsPerCurve  = 30,
                               distPerTab      = 8,
                               tabWidth        = 0.25,
                               cutterDiameter  = cutterDiameter
                        )
      cncGcodeGenerator = cncGcodeGeneratorClass(cncPaths           = cncPaths,
                                           materialThickness  = materialThickness,
                                           depthBelowMaterial = 0.06,
                                           depthPerPass       = 0.1,
                                           cutFeedRate        = 79,
                                           safeHeight         = 1.0,
                                           tabHeight          = 0.12,
                                           useMM              = False # use inches
                                          )
      cncGcodeGenerator.Generate()
      self.set_inches()
      self.absolute_move(z = -0.25)
      print("SENDING GCODE")
      for gCode in cncGcodeGenerator.gCodes:
          print("CODE: " + str(gCode))
          self.gerbil.stream(str(gCode) + "\n")

      self.absolute_move(z = -0.25)

    def set_inches(self):
        self.gerbil.send_immediately("G20\n")

    def set_mm(self):
        self.gerbil.send_immediately("G21\n")

    def send_file(self, gCodeFile, xOffset, yOffset, rotation):
        #Set to inches for offset and rotations
        self.set_inches()

        ##########################################
        #Offset work to desired offset
        ##########################################
        self.set_work_coord_offset(xOffset, yOffset)

        ##########################################
        #Rotate work to desired rotation
        ##########################################
        deg = -rotation * 180 / math.pi
        self.gerbil.send_immediately("G68 X0 Y0 R" + str(deg) + "\n")

        #Set back to mm, typically the units g code assumes
        self.set_mm()

        ZFound = False
        with open(gCodeFile, 'r') as fh:
          for line_text in fh.readlines():
            if " Z" in line_text.upper():
              ZFound = True
              break

        # if Z move found in file (not a laser cutting file), then move cutter away from workspace as first move
        # so that if it was forgotten to do that, the first rapid traverse does not run into the workpiece
        if ZFound:
          self.absolute_move(z = -0.25)

        with open(gCodeFile, 'r') as fh:
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
file = 'cnc13.jpg'
frame = cv2.imread(file)
img = cv2.imread(file)

#######################################################################
# Get grayscale image above threshold
#######################################################################
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

########################################
# Get aruco box information
########################################
boxes, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100))
#print("boxes")
#print(boxes)
#print("ids")
#print(ids)

pixelLoc = [None]*2
locations = [None]*2
sideRefLocToOrigPixelLoc = [None]*2
pixelsAtBed = [None]*2
refBoxes = [leftBoxRef, rightBoxRef]
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
  sideRefLocToOrigPixelLoc[i], status = cv2.findHomography(np.array(locations[i]), np.array(pixelLoc[i]))

  #############################################################
  # Draw vertical box on left and right vertical region of CNC
  #############################################################
  points = np.array([[refBoxes[i].Z,0],[bedSize.Z,0],[bedSize.Z,bedSize.Y],[refBoxes[i].Z,bedSize.Y]])
  #points = np.array([[refBoxes[i].Z,refBoxes[i].Y + 15.658499997],[bedSize.Z,refBoxes[i].Y + 15.658499997],[bedSize.Z,bedSize.Y],[refBoxes[i].Z,refBoxes[i].Y]])
  pixelsAtBed[i] = cv2.perspectiveTransform(points.reshape(-1,1,2), sideRefLocToOrigPixelLoc[i])
  display_4_lines(pixelsAtBed[i], frame)

####################################################################################################
# Get forward and backward homography from simulated overhead Pixel location to Orig pixel location
# Makes destination image same size as source image.  Reshaped later due to matplot lib speed limitations
####################################################################################################
#shape[0] is height.  shape[1] is width
#PixelCorners are [height,0], height, width
height = float(frame.shape[1])
width  = float(frame.shape[0])
bedPixelCorners = np.array([[height,0.0],[height,width],[0.0,0.0],[0.0,width]])
refPixels = np.array([pixelsAtBed[0][1],pixelsAtBed[0][2],pixelsAtBed[1][1],pixelsAtBed[1][2]])
bedPixelToOrigPixelLoc, status    = cv2.findHomography(bedPixelCorners, refPixels)
origPixelToBedPixelLoc, status    = cv2.findHomography(refPixels, bedPixelCorners)
  
#############################################################
# Draw box on CNC bed
#############################################################
pixels = cv2.perspectiveTransform(bedPixelCorners.reshape(-1,1,2), bedPixelToOrigPixelLoc)
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
cv2Overhead = cv2.warpPerspective(frame, origPixelToBedPixelLoc, (frame.shape[1], frame.shape[0]))
cv2Overhead = cv2.resize(cv2Overhead, (bedViewSizePixels, bedViewSizePixels))
GCodeOverlay = OverlayGcode(cv2Overhead, \
                            svgFile = 'puzzles2.svg', \
                            #gCodeFile = gCodeFile, \
                            enableSender = False)

########################################
# Detect box location in overhead image
########################################
#Change overhead image to gray for box detection
refPixelLoc    = get_id_loc(frame, boxes, ids, 66)
refPhysicalLoc = cv2.perspectiveTransform(refPixelLoc.reshape(-1,1,2), origPixelToBedPixelLoc)
touchPlateLocPercent = refPhysicalLoc / [frame.shape[1], frame.shape[0]]
touchPlateLoc = []
touchPlatePixels = []
for a in touchPlateLocPercent:
    touchPlateLoc.append(a[0] * [bedSize.X, bedSize.Y])
    touchPlatePixels.append(a[0] * [bedViewSizePixels, bedViewSizePixels] )
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("Touch Plate Loc: " + str(touchPlateLoc))
GCodeOverlay.set_ref_loc(touchPlatePixels)

######################################################################
# Create a G Code sender now that overlay is created
######################################################################

plt.show()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
