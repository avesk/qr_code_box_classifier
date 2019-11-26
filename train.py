import cv2
import numpy as np
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import random
from sklearn import svm
from sklearn import tree


def filter_contour(contour):
    curve_len = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.05 * curve_len, True)
    if len(approx) >= 2 and len(approx) < 50:
        x, y, w, h = cv2.boundingRect(approx)
        if h > 20 and h < 1900:
            return (x, y)

    return (-1, -1)


def feature_extract(cls, directory, documents):
    row_len = 216
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            doc = [0.0, 0.0, 0.0]
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
                card = len(areas)
                doc[0] = mean
                doc[1] = std
                doc[2] = float(card)
            documents[cls].append(doc)
    return documents


def shuffle_data(a, b):
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    return np.array(a), np.array(b)


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


def train(X, Y, clsf):
    splits = 10
    kf = KFold(n_splits=splits)

    score = 0
    conf_matrix = np.zeros((2, 2))
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # classifier
        clsf.fit(X_train, Y_train)
        score += clsf.score(X_test, Y_test)
        Y_pred = clsf.predict(X_test)
        conf_matrix += confusion_matrix(Y_test, Y_pred)
    return score / splits, conf_matrix


X, Y = prep_data()
X, Y = shuffle_data(X, Y)

#prints out data for csv file
for i in range(len(X)):
    for j in range(len(X[i])):
        print(X[i][j], end=", ")
    if Y[i] == 0.0:
        print("yes")
    else:
        print("no")

# Naive Bayes
score, conf_matrix = train(X, Y, GaussianNB())

print("GNB")
print(score)
print(conf_matrix)

#Logistic Regression
score, conf_matrix = train(X, Y, LogisticRegression(solver= 'liblinear' ))

print("Logistic")
print(score)
print(conf_matrix)

# SVM
score, conf_matrix = train(X, Y, svm.SVC(gamma='scale'))

print("SVM")
print(score)
print(conf_matrix)


# Decision Tree
score, conf_matrix = train(X, Y, tree.DecisionTreeClassifier())

print("Tree")
print(score)
print(conf_matrix)
