import numpy as np
import cv2


def find_features(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 7, 0.01, 10)

    if corners is not None:
        corners = np.int0(corners)
        for i in corners:
            x,y = i.ravel()
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

def contours(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Color thresholding
    ret, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)

    # Find the contours of the frame
    contours, hierarchy = cv2.findContours(thresh.copy(), 1, cv2.CHAIN_APPROX_NONE)

    return contours

def boundingRect(img):
    cnts = contours(img)
    if len(cnts) <= 0:
        return

    cnt = max(cnts, key=cv2.contourArea)
    if cnt is None:
        return

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

def hough_lines(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the edges in the image using canny detector
    edges = cv2.Canny(gray, 50, 200)

    # Detect points that form a line
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=10, maxLineGap=250)

    # Draw lines on the image
    if lines is not None and len(lines > 0):
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

def contours(img):
    # img = (255 - img)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Color thresholding
    ret, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)

    # Find the contours of the frame
    contours, hierarchy = cv2.findContours(thresh.copy(), 1, cv2.CHAIN_APPROX_SIMPLE)

    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print('...................')
    print(len(contours))
    print('-------------------')

    # Find the biggest contour (if detected)
    if len(contours) > 0:
        # c = max(contours, key=cv2.contourArea)
        # M = cv2.moments(c)
        #
        # cx = int(M['m10']/M['m00'])
        # cy = int(M['m01']/M['m00'])
        #
        # cv2.line(img,(cx,0),(cx,720),(0, 0, 255),1)
        # cv2.line(img,(0,cy),(1280,cy),(0, 0, 255),1)
        cv2.drawContours(img, contours, 1, (0, 255, 0), 3)

    #     if cx >= 120:
    #         print("Turn Left!")
    #
    #     if cx < 120 and cx > 50:
    #         print("On Track!")
    #
    #     if cx <= 50:
    #         print("Turn Right")
    #
    # else:
    #     print("I don't see the line")


# video_capture = cv2.VideoCapture(0)
videoCapture = cv2.VideoCapture('http://192.168.4.92:8080/video')
if not videoCapture.isOpened():
    raise Exception('Couldn\'t establish video connection')

while(True):
    # Capture the frames
    ret, img = videoCapture.read()
    if img is None:
        print('no frame')
        continue

    # find_features(img)
    # boundingRect(img)
    contours(img)

    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
