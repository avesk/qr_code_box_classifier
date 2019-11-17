import cv2
import numpy as np
import imutils
import os
from sklearn.naive_bayes import GaussianNB


def filter_contour(contour):
    curve_len = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.05 * curve_len, True)
    if len(approx) >= 2 and len(approx) < 50:
        x, y, w, h = cv2.boundingRect(approx)
        if h > 5 and h < 1900:
            return (x, y)

    return (-1, -1)


def coord_to_index(x, y, row_len):
    return float(x*row_len + y)


def flatten_contour(contour, row_len):
    flattened_contour = []
    for cont in contour:
        x, y = cont[0][0], cont[0][1]
        flattened_contour.append(coord_to_index(x, y, row_len))
    return flattened_contour


def feature_extract(cls, directory, documents):
    row_len = 216
    for filename in os.listdir(directory):
        doc = []
        frame = cv2.imread(f"{directory}/{filename}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

        contours, h = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cont in contours:
            x, y = filter_contour(cont)
            if x > -1:
                doc += flatten_contour(cont, row_len)
        documents[cls].append(doc)
    return documents


documents = {"pos": [], "neg": []}

# Positives
directory = "./training-data/contains_qr_code"
documents = feature_extract("pos", directory, documents)

# Negatives
directory = "./training-data/no_qr_code"
documents = feature_extract("neg", directory, documents)

pos_x = np.asarray(documents["pos"])
neg_x = np.asarray(documents["neg"])
# print(pos_x[0])

# print(len(pos_x))
# print(len(neg_x))

X = np.concatenate((pos_x, neg_x))
Y = np.concatenate((np.zeros(len(pos_x)), np.ones(len(neg_x))))

# print(len(X))
# print(len(Y))
# print(Y)
# score = 0
gnb = GaussianNB()
gnb.fit(X, Y)
# score += gnb.score(X_test, Y_test)
# Y_pred = gnb.predict(X[0])

# print(Y_pred)
