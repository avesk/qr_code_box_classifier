import cv2
import numpy as np
import imutils


def create_window(win_name):
    # Create image window
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 600, 600)


def display_lines(frame, lines):
    if lines is not None:
        for line in lines[:100]:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(img=frame, pt1=(x1, y1), pt2=(
                x2, y2), color=(255, 0, 0), thickness=3)
    return frame


def detect_shape(contour):
    curve_len = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.05 * curve_len, True)
    ar_tol = .1
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ar = w / float(h)
        if ar >= 1.0-ar_tol and ar <= 1.0+ar_tol and h > 5:
            print('found square')
            return (x, y)
    return -1, -1


win_name = 'frame'
create_window(win_name)

filepath = 'training-data/4.jpg'


frame = cv2.imread(filepath)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(
    src=gray, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY, blockSize=3, C=5
)

contours, h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cont in contours:
    x, y = detect_shape(cont)
    if x > -1:
        print(cont)
        cv2.drawContours(frame, [cont], 0, (0, 255, 0), 2)
        print(x, y)

# cv2.imshow(win_name, thresh)
cv2.imshow(win_name, frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
