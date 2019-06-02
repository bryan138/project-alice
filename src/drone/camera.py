import numpy as np
import cv2
import random
from math import atan2, cos, sin, sqrt, pi, radians
from centroidtracker import CentroidTracker


SOURCE = 2 # 0 - Stream, 1 - Photo, 2 - Video, default - Webcam

ARROW_MATCH_THRESHOLD = 0.1
CONTOUR_AREA_FILTER = (800, 15000)


def goodFeatures(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 7, 0.01, 10)
    if corners is not None:
        corners = np.int0(corners)
        for i in corners:
            x, y = i.ravel()
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

def boundingRect(img):
    contours = getContours(img)
    if len(contours) <= 0:
        return

    maxContour = max(contours, key=cv2.contourArea)
    if maxContour is None:
        return

    rect = cv2.minAreaRect(maxContour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

def houghLines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=10, maxLineGap=250)

    # Draw lines on the image
    if lines is not None and len(lines > 0):
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

def getContours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(blur, 128, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def drawContours(img):
    contours = getContours(img)

    if len(contours) > 0:
        # Find the largest contour and draw its centroid
        largestContour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largestContour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.line(img, (cx, 0), (cx, 720), (255, 0, 0), 1)
        cv2.line(img, (0, cy), (1280, cy), (255, 0, 0), 1)

        cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

    return contours

def contourOrientation(arrows, img):
    for arrow in arrows:
        center, (MA, ma), angle = cv2.fitEllipse(arrow)
        angle = radians(angle - 90)
        x = center[0] + MA * cos(angle)
        y = center[1] + MA * sin(angle)
        drawAxis(img, center, (x, y), (0, 255, 0), 1)

def getPointSide(p, p1, p2):
    return (p[0] - p1[0]) * (p2[1] - p1[1]) - (p[1] - p1[1]) * (p2[0] - p1[0])

def rotatePoint(point, reference, angle):
    rotatedPoint = [0, 0]
    rotatedPoint[0] = (point[0] - reference[0]) * cos(angle) - (point[1] - reference[1]) * sin(angle) + reference[0]
    rotatedPoint[1] = (point[0] - reference[0]) * sin(angle) + (point[1] - reference[1]) * cos(angle) + reference[1]
    return rotatedPoint

def pcaOrientation(arrows, img):
    for arrow in arrows:
        # Construct a buffer used by the PCA analysis
        size = len(arrow)
        dataPoints = np.empty((size, 2), dtype=np.float64)
        for i in range(dataPoints.shape[0]):
            dataPoints[i, 0] = arrow[i, 0, 0]
            dataPoints[i, 1] = arrow[i, 0, 1]

        # Perform PCA analysis
        mean = np.empty((0))
        mean, eigenVectors, eigenValues = cv2.PCACompute2(dataPoints, mean)
        center = (int(mean[0, 0]), int(mean[0, 1]))
        majorAxis = (center[0] + 0.02 * eigenVectors[0, 0] * eigenValues[0, 0], center[1] + 0.02 * eigenVectors[0, 1] * eigenValues[0, 0])
        minorAxis = (center[0] - 0.02 * eigenVectors[1, 0] * eigenValues[1, 0], center[1] - 0.02 * eigenVectors[1, 1] * eigenValues[1, 0])

        # Count points in each side of the minor axis
        pointCount = [0, 0]
        for component in arrow:
            point = (component[0, 0], component[0, 1])
            side = getPointSide(point, center, minorAxis)
            if side < 0:
                pointCount[0] += 1
            else:
                pointCount[1] += 1

        # Correct orientation of major axis, if necessary
        arrowOrientation = -1 if pointCount[0] > pointCount[1] else 1
        if np.sign(arrowOrientation) != np.sign(getPointSide(majorAxis, center, minorAxis)):
            majorAxis = rotatePoint(majorAxis, center, pi)

        # Draw the principal components
        cv2.circle(img, center, 3, (255, 0, 255), 1)
        drawAxis(img, center, majorAxis, (0, 255, 255), 1)
        drawAxis(img, center, minorAxis, (255, 255, 0), 2)

        # angle = atan2(eigenVectors[0, 1], eigenVectors[0, 0])
        # return angle

def drawAxis(img, p, q, colour, scale):
    p = list(p)
    q = list(q)

    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

    # Draw arrow axis
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)

    # Draw arrow head hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)

def randomColor():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def filterArrows(img):
    contours = getContours(img)

    arrows = []
    otherContours = []
    for contour in contours:
        # Ignore contours that are too small or too large
        area = cv2.contourArea(contour)
        if area < CONTOUR_AREA_FILTER[0] or CONTOUR_AREA_FILTER[1] < area:
            continue

        matches = cv2.matchShapes(contour, arrowContour, cv2.CONTOURS_MATCH_I3, 0)
        if matches < ARROW_MATCH_THRESHOLD:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            arrow = cv2.approxPolyDP(contour, epsilon, True)
            arrows.append(arrow)
        else:
            otherContours.append(contour)

    return arrows, otherContours

def putText(text, contour, img):
    (x, y), radius = cv2.minEnclosingCircle(contour)
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

def tracker(arrows, img):
    rects = []

    for arrow in arrows:
		# Compute the bounding boxes for each arrow
        startX, startY, width, height = cv2.boundingRect(arrow)
        endX = startX + width
        endY = startY + height
        rects.append((startX, startY, endX, endY))

		# Draw bounding boxes for each arrow
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)

    center = (img.shape[1] / 2, img.shape[0] / 2)
    centermostCentroid = -1
    centermostDistance = float('inf')

    objects = centroidTracker.update(rects)
    for (objectID, centroid) in objects.items():
        # Draw ID and centroid of arrows
        text = "ID {}".format(objectID)
        cv2.putText(img, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(img, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        distanceFromCenter = abs(center[0] - centroid[0]) + abs(center[1] - centroid[1])
        if distanceFromCenter < centermostDistance:
            centermostDistance = distanceFromCenter
            centermostCentroid = centroid

    if centermostCentroid is not -1:
        # Look for contour with the centermost centroid
        for arrow in arrows:
            startX, startY, width, height = cv2.boundingRect(arrow)
            endX = startX + width
            endY = startY + height

            arrowCenterX = int((startX + endX) / 2.0)
            arrowCenterY = int((startY + endY) / 2.0)

            if centermostCentroid[0] == arrowCenterX and centermostCentroid[1] == arrowCenterY:
                cv2.drawContours(img, [arrow], -1, (255, 255, 0), 3)

if SOURCE == 0:
    videoCapture = cv2.VideoCapture('http://192.168.0.118:8080/video')
elif SOURCE == 1:
    videoCapture = cv2.VideoCapture('assets/arrow_photo.jpg')
elif SOURCE == 2:
    videoCapture = cv2.VideoCapture('assets/arrow_video.mp4')
else:
    videoCapture = cv2.VideoCapture(0)
    videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, 360)
    videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

arrowContour = getContours(cv2.imread('assets/arrow.png'))[0]

centroidTracker = CentroidTracker()

while True:
    if SOURCE == 1:
        img = cv2.imread('assets/arrow_photo.jpg')
        img = cv2.resize(img, (562, 421))
    else:
        ret, img = videoCapture.read()

    if img is None:
        if SOURCE == 2:
            # Loop video
            videoCapture = cv2.VideoCapture('assets/arrow_video.mp4')
            ret, img = videoCapture.read()
        else:
            if not videoCapture.isOpened():
                raise Exception('Couldn\'t establish video connection')
            raise Exception('No frame found')

    if SOURCE == 2:
        img = cv2.resize(img, (480, 270))

    arrows, contours = filterArrows(img)
    cv2.drawContours(img, arrows, -1, (0, 0, 255), 2)
    cv2.drawContours(img, contours, -1, (255, 0, 0), 1)

    # goodFeatures(img)
    # boundingRect(img)
    # houghLines(img)
    # drawContours(img)
    # contourOrientation(arrows, img)
    pcaOrientation(arrows, img)
    # filterArrows(img)
    # tracker(arrows, img)

    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
