import cv2
import numpy as np
import imutils


def create_window(win_name):
    # Create image window
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 600, 600)


def filter_contour(contour):
    curve_len = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.05 * curve_len, True)
    if len(approx) >= 2 and len(approx) < 50:
        x, y, w, h = cv2.boundingRect(approx)
        if h > 5 and h < 1900:
            return (x, y)

    return (-1, -1)


win_name = 'frame'
create_window(win_name)

filepath = 'training-data/no_qr_code/img27_648_648.jpg'


frame = cv2.imread(filepath)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

contours, h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cont in contours:
    x, y = filter_contour(cont)
    if x > -1:
        print(cont)
        cv2.drawContours(frame, [cont], 0, (0, 255, 0), 2)
        print(x, y)
cv2.imshow(win_name, frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
