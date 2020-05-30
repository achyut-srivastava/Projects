import cv2
import numpy as np
import time
from numpy import linalg as LA
import math
global num
num= []
global zero_to_one
zero_to_one=[]
global n1
lr=0
n1=[0,0]
cap = cv2.VideoCapture(0)
def get_contour(roi):
    contours, _ = cv2.findContours(roi.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        return cnt
    else:
        return None

def finger_count(img, cnt, counter):
    global lr
    convex_hull = cv2.convexHull(cnt)
    top = tuple(convex_hull[convex_hull[:, :, 1].argmin()][0])
    bottom = tuple(convex_hull[convex_hull[:, :, 1].argmax()][0])
    left = tuple(convex_hull[convex_hull[:, :, 0].argmin()][0])
    right = tuple(convex_hull[convex_hull[:, :, 0].argmax()][0])
    cx = (left[0] + right[0]) // 2
    cy = (top[1] + bottom[1]) // 2
    #cv2.circle(img, (cx,cy), 65, [255, 255, 255], 1)
    convex_hull = cv2.convexHull(cnt,returnPoints=False)
    defects = cv2.convexityDefects(cnt, convex_hull)

    if defects is not None:
        for i in range(defects.shape[0]-1):
            s, e, f, d = defects[i, 0]
            _,_,f_1,_ = defects[i+1, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            far_1 = tuple(cnt[f_1][0])
            a = np.linalg.norm(np.array(far_1)-np.array(far))
            b = np.linalg.norm(np.array(far)-np.array(end))
            c = np.linalg.norm(np.array(far_1) - np.array(end))
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c))*(180/math.pi)
            check = np.linalg.norm((cx,cy)-np.array(start))
            if angle<135 and  d>2900 and check>62:
                counter +=1
                #cv2.line(img, start, far, (0, 140, 255), 2)
                #cv2.line(img, end, far, (0, 140, 255), 2)
                #cv2.circle(img, (start[0], start[1]), 5, [255, 0,0], -1)
        if counter != 0:
            num.append(counter)
            zero_to_one.append(1)
        if counter == 0:
            zero_to_one.append(0)
        if len(zero_to_one)>2 and (zero_to_one[-1]- zero_to_one[-2])==-1:
             lr += 1
             if lr%2 ==1:
                 n1[0] = num[-1]
                 n1[1]= 0
             else:
                 n1[1] = num[-2]
    return

def preProcessing(frame):
    counter=0
    x1 = 10
    x2 = 210
    y2 = 270
    y1 = 70
    cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 0, 255), 2)
    roi = frame[y1:y2, x1:x2]
    img = frame[70:480, 10:640]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi = cv2.threshold(roi, 156, 255, cv2.THRESH_BINARY_INV)
    roi = cv2.medianBlur(roi,7)
    cnt_max = get_contour(roi)
    if cnt_max is not None:
        cv2.drawContours(frame, [cnt_max + (10, 70)], -1, (0, 255, 0), 2)
        counter = finger_count(img,cnt_max, counter)
    cv2.imshow("roi", roi)
    return img

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame,1)
    img =preProcessing(frame)
    if n1[0]!=0 and n1[1]==0:
        cv2.putText(frame,str(n1[0]),(210,50),cv2.FONT_HERSHEY_PLAIN, 3,(0,255,0),2)
    if n1[1]!=0 and n1[1]!=0:
        cv2.putText(frame,str(n1[0])+" + "+ str(n1[1])+" = "+ str(n1[0]+n1[1]),(210,50),cv2.FONT_HERSHEY_PLAIN, 3,(0,255,0),2)
    cv2.imshow("Image", frame)
    key = cv2.waitKey(10)
    if key & 0xFF == ord('q'):
        break