import cv2
import os
import re

def getFileName(file):
    m = re.search('(.*).jpg')
    if m:
        return m.group(1)
    return f"{math.random()}error"

directory = os.fsencode("/training-data")
for file in os.listdir(directory):
    newDir = f"{getFileName(file)}"
    os.mkdir(newDir)
    img = cv2.imread(file)
    for r in range(0,img.shape[0],30):
        for c in range(0,img.shape[1],30):
            cv2.imwrite(f"{newDir}/img{r}_{c}.png",img[r:r+30, c:c+30,:])