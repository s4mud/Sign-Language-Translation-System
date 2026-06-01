import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier(
    r"F:\new baas\SLRNet-main\HandSignDetector\Model\keras_model.h5",
    r"F:\new baas\SLRNet-main\HandSignDetector\Model\labels.txt"
)

offset = 20
imgSize = 300

folder = "Images/C"
counter = 0

labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","v","W","X","Y",
          "Z","HELLO","PLEASE","SORRY","THANKS","WELCOME","HOW","WELCOME","NICE","HELP","LISTEN"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop = img[y-offset:y + h+offset, x-offset:x + w+offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize/h
            wCal=math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal,imgSize))
            imgResizeShape= imgResize.shape
            wGap=math.ceil((300-wCal)/2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)


        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((300 - hCal) / 2)
            imgWhite[hGap:hCal + hGap,:] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.putText(imgOutput, labels[index], (x,y-26), cv2.FONT_HERSHEY_TRIPLEX,1.1,(0,0,255), 3)
        cv2.rectangle(imgOutput, (x-offset,y-offset),(x + w+ offset, y + h + offset),(0,255,0), 4)

        cv2.imshow("ImageCrop",imgCrop)
        cv2.imshow("imgWhite",imgWhite)

        cv2.imshow("image",imgOutput)
        cv2.waitKey(1)