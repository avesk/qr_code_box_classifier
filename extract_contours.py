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


def feature_extract(cls, directory, documents):
    row_len = 216
    for filename in os.listdir(directory):
        doc = [0.0, 0.0]
        frame = cv2.imread(f"{directory}/{filename}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

        contours, h = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        areas = []
        for i, cont in enumerate(contours):
            x, y = filter_contour(cont)
            if x > -1:
                areas.append(cv2.contourArea(cont))
        if len(areas) > 0:
            mean = np.mean(areas)
            std = np.std(areas)
            doc[0] = mean
            doc[1] = std
        documents[cls].append(doc)
    return documents


def prep_data():
    documents = {"pos": [], "neg": []}

    # Positives
    directory = "./training-data/contains_qr_code"
    documents = feature_extract("pos", directory, documents)

    # Negatives
    directory = "./training-data/no_qr_code"
    documents = feature_extract("neg", directory, documents)

    pos_x = np.asarray(documents["pos"])
    neg_x = np.asarray(documents["neg"])

    X = np.concatenate((pos_x, neg_x))
    Y = np.concatenate((np.zeros(len(pos_x)), np.ones(len(neg_x))))

    return X, Y


def classify(X, Y, alg):
    score = 0
    clsf = alg()
    clsf.fit(X, Y)
    score += clsf.score(X, Y)
    return score


X, Y = prep_data()
score = classify(X, Y, GaussianNB)

print(score)
