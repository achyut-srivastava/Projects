import cv2
import numpy as np
import os

if not os.path.exists('data'):
    os.makedirs("data")
    os.makedirs("data/0")
    os.makedirs("data/1")
    os.makedirs("data/2")
    os.makedirs("data/3")
    os.makedirs("data/4")
    os.makedirs("data/5")

path = 'data'
camera=0
cap = cv2.VideoCapture(camera)
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame,1)
    count = {
        'zero': len(os.listdir(path + "/0")),
        'one': len(os.listdir(path + "/1")),
        'two': len(os.listdir(path + "/2")),
        'three': len(os.listdir(path + "/3")),
        'four': len(os.listdir(path + "/4")),
        'five': len(os.listdir(path + "/5")),
    }
    total_img = count['zero']+count['one']+count['three']+count['four']+count['five']+count['two']
    cv2.putText(frame,"Images :"+str(total_img),(10,50),cv2.FONT_HERSHEY_PLAIN, 1.5,(0,255,0),1)
    cv2.putText(frame, "0 : " + str(count['zero']), (10, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)
    cv2.putText(frame, "1 : " + str(count['one']), (10, 110), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)
    cv2.putText(frame, "2 : " + str(count['two']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)
    cv2.putText(frame, "3 : " + str(count['three']), (10, 170), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)
    cv2.putText(frame, "4 : " + str(count['four']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)
    cv2.putText(frame, "5 : " + str(count['five']), (10, 230), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)

    x1 = int(0.5*frame.shape[1])
    x2 = frame.shape[1]-10
    y2 = int(0.5 * frame.shape[1])
    y1 =10
    cv2.rectangle(frame, (x1,y1),(x2,y2),(0,0,255),2)
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi,(64,64))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi = cv2.threshold(roi,190,255,cv2.THRESH_BINARY)
    cv2.imshow("Roi", roi)

    cv2.imshow("Image", frame)

    key = cv2.waitKey(10)
    if key & 0xFF == 27:
        break
    if key & 0xFF == ord('0'):
        cv2.imwrite(path+"/0/"+str(count['zero'])+".jpg", roi)
    if key & 0xFF == ord('1'):
        cv2.imwrite(path + "/1/" + str(count['one']) + ".jpg", roi)
    if key & 0xFF == ord('2'):
        cv2.imwrite(path + "/2/" + str(count['two']) + ".jpg", roi)
    if key & 0xFF == ord('3'):
        cv2.imwrite(path + "/3/" + str(count['three']) + ".jpg", roi)
    if key & 0xFF == ord('4'):
        cv2.imwrite(path + "/4/" + str(count['four']) + ".jpg", roi)
    if key & 0xFF == ord('5'):
        cv2.imwrite(path + "/5/" + str(count['five']) + ".jpg", roi)
        
cap.release()
cv2.destroyAllWindows()




