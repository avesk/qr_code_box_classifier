import cv2
import os
import re


def getFileName(fname):
    m = re.search('(.*).jpg', fname)
    if m:
        return m.group(1)
    return "error"


ar = 216
directory = os.fsencode("./training-data").decode()
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        newDir = f"{getFileName(filename)}"
        img = cv2.imread(f"{directory}/{filename}")
        os.mkdir(f"./{directory}/{newDir}")
        for r in range(0, img.shape[0], ar):
            for c in range(0, img.shape[1], ar):
                cv2.imwrite(f"./{directory}/{newDir}/img{newDir}_{r}_{c}.jpg",
                            img[r:r+ar, c:c+ar, :])
