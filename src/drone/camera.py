import numpy as np
import cv2
import random
import time
from math import atan2, cos, sin, sqrt, pi, radians, degrees
from centroidtracker import CentroidTracker
from djitellopy import Tello


SOURCE = 0 # 0 - Stream, 1 - Photo, 2 - Video, 3 - Loop Video, 4 - Drone, 5 - Drone Video, default - Webcam
DRONE_IS_ACTIVE = SOURCE == 4
IP_WEBCAM = 'http://192.168.0.110:8080/video'

ARROW_MATCH_THRESHOLD = 0.15
CONTOUR_AREA_FILTER = (3000, 30000)
MAX_JUMP_DISTANCE = 500

LOOKOUT_AREA_HEIGHT = 50
LOOKOUT_AREA_WIDTH = 1000
TARGET_RADIUS = 150

MOVE_STEP = 20
PROXIMITY_RANGE = [7000, 20000]

HYPOTENUSE = 30
GO_XYZ_SPEED = 20

COMMAND_INTERVAL = 0.75


class Arrow:
    def __init__(self, id, centroid, distanceFromCenter):
        self.id = id
        self.centroid = centroid
        self.distanceFromCenter = distanceFromCenter
        self.contour = None

    def __str__(self):
        return '{}: ({}, {}) - {}'.format(self.id, self.centroid[0], self.centroid[1], 'None' if self.contour is None else 'Contour')

    def __repr__(self):
        return '{}: ({}, {}) - {}'.format(self.id, self.centroid[0], self.centroid[1], 'None' if self.contour is None else 'Contour')


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
    if DRONE_IS_ACTIVE:
        ret, thresh = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY_INV)
    else:
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
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

def pcaOrientation(arrow, img):
    # Construct a buffer used by the PCA analysis
    size = len(arrow)
    dataPoints = np.empty((size, 2), dtype=np.float64)
    for i in range(dataPoints.shape[0]):
        dataPoints[i, 0] = arrow[i, 0, 0]
        dataPoints[i, 1] = arrow[i, 0, 1]

    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenVectors, eigenValues = cv2.PCACompute2(dataPoints, mean)
    center = getTuplePoint(mean[0, :])
    majorAxis = (center[0] + 0.02 * eigenVectors[0, 0] * eigenValues[0, 0], center[1] + 0.02 * eigenVectors[0, 1] * eigenValues[0, 0])
    minorAxis = (center[0] - 0.02 * eigenVectors[1, 0] * eigenValues[1, 0], center[1] - 0.02 * eigenVectors[1, 1] * eigenValues[1, 0])
    angle = atan2(eigenVectors[0, 1], eigenVectors[0, 0])

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
        angle += pi

    # Draw the principal components
    cv2.circle(img, center, 3, (255, 0, 255), 1)
    drawAxis(img, center, majorAxis, (0, 255, 255), 1)
    drawAxis(img, center, minorAxis, (255, 255, 0), 2)

    return (angle, center)

def getLookoutArea(angle, center):
    pointA = np.array(center)
    pointA[0] = pointA[0] + cos(angle - pi / 2) * LOOKOUT_AREA_HEIGHT
    pointA[1] = pointA[1] + sin(angle - pi / 2) * LOOKOUT_AREA_HEIGHT
    pointB = np.array(center)
    pointB[0] = pointB[0] - cos(angle - pi / 2) * LOOKOUT_AREA_HEIGHT
    pointB[1] = pointB[1] - sin(angle - pi / 2) * LOOKOUT_AREA_HEIGHT
    pointC = np.array(pointA)
    pointC[0] = pointC[0] + cos(angle) * LOOKOUT_AREA_WIDTH
    pointC[1] = pointC[1] + sin(angle) * LOOKOUT_AREA_WIDTH
    pointD = np.array(pointB)
    pointD[0] = pointD[0] + cos(angle) * LOOKOUT_AREA_WIDTH
    pointD[1] = pointD[1] + sin(angle) * LOOKOUT_AREA_WIDTH

    return np.array([pointA, pointB, pointD, pointC], np.int32)

def drawAxis(img, p, q, colour, scale):
    p = list(p)
    q = list(q)

    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

    # Draw arrow axis
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, getTuplePoint(p), getTuplePoint(q), colour, 1, cv2.LINE_AA)

    # Draw arrow head hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, getTuplePoint(p), getTuplePoint(q), colour, 1, cv2.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, getTuplePoint(p), getTuplePoint(q), colour, 1, cv2.LINE_AA)

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
            # putText('{0:.2f} {1:.2f}'.format(area, matches), arrow, img)
        else:
            otherContours.append(contour)

    return arrows, otherContours

def putText(text, contour, img):
    center, radius = cv2.minEnclosingCircle(contour)
    center = (center[0] + 75, center[1] - 75)
    cv2.putText(img, text, getTuplePoint(center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

def getBoundingBox(contour):
    startX, startY, width, height = cv2.boundingRect(contour)
    endX = startX + width
    endY = startY + height
    return (startX, startY, endX, endY)

def getCentroid(contour):
    (startX, startY, endX, endY) = getBoundingBox(contour)
    centerX = int((startX + endX) / 2.0)
    centerY = int((startY + endY) / 2.0)
    return (centerX, centerY)

def getTuplePoint(point):
    return (int(point[0]), int(point[1]))

def moveDrone(direction, value=MOVE_STEP):
    global lastCommand
    currentTime = time.time()

    if currentTime - lastCommand > COMMAND_INTERVAL and flightActivated and drone is not None:
        lastCommand = currentTime
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>MOVE:', direction)
        drone.move(direction, value)

def goToAngle(angle):
    global lastCommand
    currentTime = time.time()

    if currentTime - lastCommand > COMMAND_INTERVAL and flightActivated and drone is not None:
        radians = angle * pi / 180
        x = int(HYPOTENUSE * cos(radians))
        y = int(HYPOTENUSE * sin(radians))

        lastCommand = currentTime
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>GO TO:', angle, x, y)
        drone.go_xyz_speed(0, x, y, GO_XYZ_SPEED)

def centerArrow(arrow, point):
    deltaX = arrow.centroid[0] - point[0]
    deltaY = point[1] - arrow.centroid[1]

    if abs(deltaX) > TARGET_RADIUS / 2:
        direction = 'right' if deltaX > 0 else 'left'
        moveDrone(direction)
        return False

    if abs(deltaY) > TARGET_RADIUS / 2:
        direction = 'up' if deltaY > 0 else 'down'
        moveDrone(direction)
        return False

    print('---------------------------CENTERED')
    return True

def adjustProximity(arrow):
    area = cv2.contourArea(arrow.contour)
    if area < PROXIMITY_RANGE[0]:
        moveDrone('forward')
        return False
    elif area > PROXIMITY_RANGE[1]:
        moveDrone('back')
        return False

    print('---------------------------NEAR')
    return True

def tracker(arrowContours, img):
    rects = []

    for arrowContour in arrowContours:
		# Compute the bounding boxes for each arrow
        (startX, startY, endX, endY) = getBoundingBox(arrowContour)
        rects.append((startX, startY, endX, endY))

		# Draw bounding boxes for each arrow
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 1)

    # Track arrow contours
    center = [img.shape[1] / 2, img.shape[0] / 2]
    objects = centroidTracker.update(rects)

    global activeArrowID
    global lookoutArea

    lostActiveArrow = True
    trackedArrows = {}
    arrows = []
    for (objectID, centroid) in objects.items():
        # Draw ID and centroid of arrows
        text = "ID {}".format(objectID)
        cv2.putText(img, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.circle(img, getTuplePoint(centroid), 2, (255, 255, 255), -1)

        # Create arrow objects
        distanceFromCenter = abs(center[0] - centroid[0]) + abs(center[1] - centroid[1])
        arrow = Arrow(objectID, centroid, distanceFromCenter)
        arrows.append(arrow)
        trackedArrows[arrow.id] = arrow

        if objectID == activeArrowID:
            lostActiveArrow = False

        # Match it to its contour, if any
        for arrowContour in arrowContours:
            arrowCenterX, arrowCenterY = getCentroid(arrowContour)
            if centroid[0] == arrowCenterX and centroid[1] == arrowCenterY:
                arrow.contour = arrowContour

    if lostActiveArrow:
        activeArrowID = -1

    # Sort arrows by center proximity
    arrows = sorted(arrows, key=lambda arrow: arrow.distanceFromCenter)

    # Draw center target area
    cv2.circle(img, getTuplePoint(center), TARGET_RADIUS, (255, 255, 255), 0)

    if activeArrowID != -1:
        activeArrow = trackedArrows[activeArrowID]
        cv2.circle(img, getTuplePoint(activeArrow.centroid), 13, (255, 255, 255), -1)

        if activeArrow.contour is not None:
            # Perform PCA analysis and draw lookout area
            angle, pcaCenter = pcaOrientation(activeArrow.contour, img)
            transformedAngle = degrees(2 * pi - angle) % 360
            # putText(str(transformedAngle), activeArrow.contour, img)
            lookoutArea = getLookoutArea(angle, pcaCenter)

    if flightActivated:
        cv2.circle(img, (25, 25), 15, (0, 255, 0), -1)

    centermostArrow = None
    for arrow in arrows:
        if arrow.contour is not None:
            centermostArrow = arrow
            break

    if centermostArrow is not None:
        cv2.drawContours(img, [centermostArrow.contour], -1, (244, 66, 170), -1)
        near = adjustProximity(centermostArrow)
        centered = centerArrow(centermostArrow, center)

    if len(arrows) > 0:
        # Draw centermost arrow, if any
        if arrows[0].contour is not None:
            cv2.drawContours(img, [arrows[0].contour], -1, (255, 255, 0), 3)

        # Mark first arrow as active if tracking hasn't started
        if activeArrowID == -1 and lookoutArea is None:
            activeArrowID = arrows[0].id

    if lookoutArea is not None:
        cv2.drawContours(img, [lookoutArea], -1, (0, 255, 255), 1)

        # TODO: Flight in arrows direction

        for arrow in arrows:
            if arrow.id != activeArrowID:
                inside = cv2.pointPolygonTest(lookoutArea, getTuplePoint(arrow.centroid), False)

                if inside >= 0:
                    # Arrow is inside the lookout area, mark it as active
                    cv2.circle(img, getTuplePoint(arrow.centroid), 5, (255, 255, 0), -1)

                    # Mark as active if arrow is inside both lookout area and target radius
                    if arrow.distanceFromCenter < TARGET_RADIUS:
                        activeArrowID = arrow.id

                else:
                    cv2.circle(img, getTuplePoint(arrow.centroid), 4, (0, 255, 0), -1)

drone = None
flightActivated = False
lastCommand = -1

sourceIsVideo = False
arrowContour = getContours(cv2.imread('assets/arrow.png'))[0]

centroidTracker = CentroidTracker()
activeArrowID = -1
lookoutArea = None

if SOURCE == 0:
    videoCapture = cv2.VideoCapture(IP_WEBCAM)
elif SOURCE == 1:
    videoCapture = cv2.VideoCapture('assets/arrow_photo.jpg')
elif SOURCE == 2 or SOURCE == 3 or SOURCE == 5:
    sourceIsVideo = True
    if SOURCE == 2:
        videoSourcePath = 'assets/arrow_video.mp4'
    elif SOURCE == 3:
        videoSourcePath = 'assets/loop.mp4'
    elif SOURCE == 5:
        videoSourcePath = 'assets/drone2.mp4'
        videoSourcePath = 'assets/proximity.mp4'
        videoSourcePath = 'assets/marco_drone.mp4'

    videoCapture = cv2.VideoCapture(videoSourcePath)
    # videoCapture.set(cv2.CAP_PROP_POS_MSEC, 75000)
elif SOURCE == 4:
    drone = Tello()

    if not drone.connect():
        raise Exception('Could not establish connection to drone')

    if not drone.set_speed(20):
        raise Exception('Not set speed to lowest possible')

    if not drone.streamoff():
        raise Exception('Could not stop video stream')

    if not drone.streamon():
        raise Exception('Could not start video stream')

    frameRead = drone.get_frame_read()

else:
    videoCapture = cv2.VideoCapture(0)
    videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, 360)
    videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    if SOURCE == 1:
        img = cv2.imread('assets/arrow_photo.jpg')
        img = cv2.resize(img, (562, 421))
    elif SOURCE == 4:
        if frameRead.stopped:
            frameRead.stop()
            raise Exception('Frame read stopped')

        img  = frameRead.frame
    else:
        ret, img = videoCapture.read()

    if img is None:
        if sourceIsVideo:
            # Loop video
            videoCapture = cv2.VideoCapture(videoSourcePath)
            ret, img = videoCapture.read()
        else:
            if not videoCapture.isOpened():
                raise Exception('Couldn\'t establish video connection')
            raise Exception('No frame found')

    if sourceIsVideo:
        img = cv2.resize(img, (int(img.shape[1] / 2.5), int(img.shape[0] // 2.5)))

    arrows, contours = filterArrows(img)
    cv2.drawContours(img, arrows, -1, (0, 0, 255), 2)
    cv2.drawContours(img, contours, -1, (255, 0, 0), 1)

    # goodFeatures(img)
    # boundingRect(img)
    # houghLines(img)
    # drawContours(img)
    # contourOrientation(arrows, img)
    # pcaOrientation(arrows, img)
    # filterArrows(img)
    tracker(arrows, img)

    cv2.imshow('frame', img)
    cv2.moveWindow('frame', 20, 40)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    elif key == ord('r'):
        activeArrowID = -1
        lookoutArea = None

    elif key == 13: # Enter
        flightActivated = not flightActivated

    elif key == 32: # Space
        flightActivated = False
        drone.land()

    elif key == ord('t'):
        drone.takeoff()

    elif key == ord('w'):
        drone.move_forward(20)

    elif key == ord('a'):
        drone.move_left(20)

    elif key == ord('s'):
        drone.move_back(20)

    elif key == ord('d'):
        drone.move_right(20)

    elif key == ord('e'):
        drone.move_up(20)

    elif key == ord('c'):
        drone.move_down(20)

    elif key == ord('j'):
        drone.rotate_clockwise(20)

    elif key == ord('l'):
        drone.rotate_counter_clockwise(20)
