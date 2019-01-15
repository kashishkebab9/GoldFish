import cv2, time
import numpy as np
import matplotlib.pyplot as plt
import imutils
from collections import deque
import argparse
import urllib

#Enter IP Address of your IP Webcam Here
xcoord = 1280
ycoord = 720
cap = cv2.VideoCapture(0)

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")

args = vars(ap.parse_args())
ocx = 0
ocy = 0

fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, img = cap.read()
    #Code for Importing Video from Phone


    fgmask = fgbg.apply(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    pts = deque(maxlen=args["buffer"])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(img,img, mask= mask)

    """cv2.imshow('original', frame)"""

    red_color = (0, 0, 255)
    green_color = (0, 255, 0)

    lineColor1 = red_color
    lineColor2 = red_color
    lineColor3 = red_color
    lineColor4 = red_color
    blur = cv2.blur(res, (3,3))

    cv2.line(res, pt1=(xcoord/2, 0), pt2=(xcoord/2, ycoord/2), color=lineColor1, thickness=2, lineType=8, shift=0)
    cv2.line(res, pt1=(0, ycoord/2), pt2=(xcoord/2, ycoord/2), color=lineColor2, thickness=2, lineType=8, shift=0)
    cv2.line(res, pt1=(xcoord/2,ycoord/2 ), pt2=(xcoord, ycoord/2), color=lineColor3, thickness=2, lineType=8, shift=0)
    cv2.line(res, pt1=(xcoord/2,ycoord/2 ), pt2=(xcoord/2, ycoord), color=lineColor4, thickness=2, lineType=8, shift=0)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    if len(cnts) > 0:
      # find the largest contour in the mask, then use
      # it to compute the minimum enclosing circle and
      # centroid

      c = max(cnts, key=cv2.contourArea)
      ((x, y), radius) = cv2.minEnclosingCircle(c)
      M = cv2.moments(c)
      center = ( int(M['m10'] / M['m00']) , int(M['m01'] / M['m00']))
      cx = center[0]
      cy = center[1]
      cv2.line(res, pt1=(0, ycoord/2), pt2=(0, cy), color=green_color, thickness=2, lineType=8, shift=0)
      cv2.line(res, pt1=(xcoord/2, 0), pt2=(cx, 0), color=green_color, thickness=2, lineType=8, shift=0)


      if cx > xcoord/2 and cy < ycoord/2:
          cv2.line(res, pt1=(xcoord/2, 0), pt2=(xcoord/2,ycoord/2 ), color=green_color, thickness=2, lineType=8, shift=0)
          cv2.line(res, pt1=(xcoord/2,ycoord/2 ), pt2=(xcoord, ycoord/2), color=green_color, thickness=2, lineType=8, shift=0)
      elif cx < xcoord/2 and cy < ycoord/2:
          cv2.line(res, pt1=(xcoord/2, 0), pt2=(xcoord/2, ycoord/2), color=green_color, thickness=2, lineType=8, shift=0)
          cv2.line(res, pt1=(0, ycoord/2), pt2=(xcoord/2, ycoord/2), color=green_color, thickness=2, lineType=8, shift=0)
      elif cx > xcoord/2 and cy > ycoord/2:
          cv2.line(res, pt1=(xcoord/2, ycoord/2), pt2=(xcoord, ycoord/2), color=green_color, thickness=2, lineType=8, shift=0)
          cv2.line(res, pt1=(xcoord/2, ycoord/2), pt2=(xcoord/2, ycoord), color=green_color, thickness=2, lineType=8, shift=0)
      elif cx < xcoord/2 and cy > ycoord/2:
          cv2.line(res, pt1=(0, ycoord/2), pt2=(xcoord/2, ycoord/2), color=green_color, thickness=2, lineType=8, shift=0)
          cv2.line(res, pt1=(xcoord/2, ycoord/2), pt2=(xcoord/2, ycoord), color=green_color, thickness=2, lineType=8, shift=0)

      # only proceed if the radius meets a minimum size
      if radius > 10:
        # draw the circle and centroid on the frame,
        # then update the list of tracked points
        cv2.circle(res, (int(x), int(y)), int(radius),
          (0, 255, 255), 2)
        cv2.circle(res, center, 5, (0, 0, 255), -1)
      pts.appendleft(center)

      if radius > 10:
          cv2.line(res,(ocx,ocy),(cx,cy),(255,255,255),2)
          ocx=cx
          ocy=cy
    for i in range(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
		if pts[i - 1] is None or pts[i] is None:
			continue
		# otherwise, compute the thickness of the line and
		# draw the connecting lines
		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		"""cv2.line(res, pts[i - 1], pts[i], (0, 0, 255), thickness)"""



    "resS = cv2.resize(res, (xcoord, ycoord))"
    imgS = cv2.resize(img, (xcoord, ycoord))
    """final = cv2.vconcat([resS, imgS])"""

    "finalS = cv2.resize(final, (xcoord, ycoord))"
    cv2.imshow("resS", res)
    cv2.imshow("imgS", imgS)
    "cv2.flip(finalS, -1)"

    "Resolution of 'res' is 1280 X 720"
    print(np.size(img, 0))
    print(np.size(img, 1))

    key = cv2.waitKey(1) & 0xFF
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
