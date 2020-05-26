import cv2
import os
import pickle
import numpy as np
camera= 0

cap = cv2.VideoCapture(camera)

pickle_in = open("all_files/model_trained.p","rb")
model = pickle.load(pickle_in)

def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,img = cv2.threshold(img, 145,255, cv2.THRESH_BINARY)
    img = cv2.medianBlur(img,3)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

while True:
    
    _, frame = cap.read()
    frame = cv2.flip(frame,1)
    x1 = int(0.7 * frame.shape[1])
    x2 = frame.shape[1] - 10
    y1 = 10
    y2 = int(0.4 * frame.shape[1])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (64, 64))
    roi = preProcessing(roi)
    # cv2.imshow("ROI", roi_1)
    roi = roi.reshape(1, 64, 64, 1)
    symbol = int(model.predict(roi).argmax(axis=1))
    predictions = model.predict(roi)
    prob = np.amax(predictions)
    if prob > 0.7:
        cv2.putText(frame,str(symbol)+ " "+ str(prob),(10,110),cv2.FONT_HERSHEY_PLAIN,5,(0,0,255),2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(10)
    if key & 0xFF == ord('q'):
        break
