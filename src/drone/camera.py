import numpy as np
import cv2
from math import atan2, cos, sin, sqrt, pi, radians


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
    ret, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
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

def contourOrientation(img):
    contours = drawContours(img)

    if len(contours) > 0:
        largestContour = max(contours, key=cv2.contourArea)
        center, (MA, ma), angle = cv2.fitEllipse(largestContour)

        angle = radians(angle - 90)
        x = center[0] + MA * cos(angle)
        y = center[1] + MA * sin(angle)
        drawAxis(img, center, (x, y), (0, 255, 0), 1)

def pcaOrientation(img):
    contours = getContours(img)

    if len(contours) > 0:
        largestContour = max(contours, key=cv2.contourArea)
        cv2.drawContours(img, [largestContour], 0, (0, 0, 255), 2);

        # Construct a buffer used by the PCA analysis
        size = len(largestContour)
        dataPoints = np.empty((size, 2), dtype=np.float64)
        for i in range(dataPoints.shape[0]):
            dataPoints[i, 0] = largestContour[i, 0, 0]
            dataPoints[i, 1] = largestContour[i, 0, 1]

        # Perform PCA analysis
        mean = np.empty((0))
        mean, eigenVectors, eigenValues = cv2.PCACompute2(dataPoints, mean)
        center = (int(mean[0, 0]), int(mean[0, 1]))

        # Draw the principal components
        cv2.circle(img, center, 3, (255, 0, 255), 2)
        p1 = (center[0] + 0.02 * eigenVectors[0, 0] * eigenValues[0, 0], center[1] + 0.02 * eigenVectors[0, 1] * eigenValues[0, 0])
        p2 = (center[0] - 0.02 * eigenVectors[1, 0] * eigenValues[1, 0], center[1] - 0.02 * eigenVectors[1, 1] * eigenValues[1, 0])
        drawAxis(img, center, p1, (0, 255, 0), 1)
        drawAxis(img, center, p2, (255, 255, 0), 2)

        angle = atan2(eigenVectors[0, 1], eigenVectors[0, 0])

        return angle

def drawAxis(img, p, q, colour, scale):
    p = list(p)
    q = list(q)

    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

    # Draw arrow axis
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 2, cv2.LINE_AA)

    # Draw arrow head hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 2, cv2.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 2, cv2.LINE_AA)


# videoCapture = cv2.VideoCapture(0)
videoCapture = cv2.VideoCapture('http://192.168.0.17:8080/video')

while True:
    # Capture the frames
    ret, img = videoCapture.read()
    if img is None:
        if not videoCapture.isOpened():
            raise Exception('Couldn\'t establish video connection')

        print('No frame found')
        continue

    # goodFeatures(img)
    # boundingRect(img)
    # houghLines(img)
    # drawContours(img)
    # contourOrientation(img)
    pcaOrientation(img)

    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
