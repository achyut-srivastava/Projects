import cv2
import numpy as np
from operator import add

W=640
H=480
CAM=0
cap= cv2.VideoCapture(CAM)
cap.set(3,W)
cap.set(4,H)

global count_frame
count_frame = 0
dig_loc = [70,110]
start = False
global hand_loc
hand_loc = [-1,-1]
x_grid=[60,115,160,210]
y_grid=[80,120,170,230,260]
buffer_digits =[0]
hand=[]
letter=""
def show_num(hand, digits_array):
    global buffer_digits
    for i in range(len(digits_array)):
        if digits_array[i][1]==(hand[1],hand[0]) and count_frame%30==0:
            buffer_digits[0] = (digits_array[i][0])
            cv2.putText(frame, str(digits_array[i][0]), (digits_array[i][2][0], digits_array[i][2][1]),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
    return buffer_digits[0]

def getHand_index(x,y):
    global hand_loc
    if count_frame%30==0:
        loc_x = np.where(x_grid > x)
        loc_y = np.where(y_grid > y)
        try:
            hand_loc[0] = loc_x[0][0] - 1
            hand_loc[1] = loc_y[0][0] - 1
        except:
            hand_loc[0] = -1
            hand_loc[1] = -1
            pass
    return hand_loc

def getTop_point(cnt_max):
    global hand
    convex_hull = cv2.convexHull(cnt_max)
    top = tuple(convex_hull[convex_hull[:,:,1].argmin()][0])
    top = list(map(add, top, (0,70)))
    M = cv2.moments(cnt_max)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    Cxy = (cx,cy)
    # cv2.circle(frame, (cx+10, cy+70), 3, (0,0,255), -1)
    # cv2.circle(frame, (top[0], top[1]), 3,(0,0,255),2)
    # print(np.linalg.norm(np.array(Cxy) - np.array(top)))
    if np.linalg.norm(np.array(Cxy)-np.array(top))> 5:
        hand = getHand_index(top[0], top[1])
        if top[0]<60:
            hand[0] = -1
            hand[1] = -1
        # cv2.putText(frame, ("Row: "+ str(hand[1])+ "  Col: "+str(hand[0])), (90, 40), cv2.FONT_HERSHEY_PLAIN,
        #         3, (255, 0, 0), 2)
    return hand

def digit_pos(count):
    digits_array=[]
    for digit_row in range(3):
        y_pos = dig_loc[1] + digit_row*55
        for digit_col in range(3):
            x_pos = dig_loc[0] + digit_col * 55
            cv2.putText(frame, chr(49+count), (x_pos, y_pos), cv2.FONT_HERSHEY_PLAIN,
                        3,(255,0,0),2)
            digits_array.append((chr(49+count),(digit_row,digit_col),(x_pos,y_pos)))
            count += 1
    cv2.putText(frame, chr(48), (x_pos//2 + 35, y_pos+55), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 2)
    digits_array.append((chr(48), (3, 1), (x_pos//2 + 35, y_pos+55)))
    # print(digits_array)
    return digits_array

def get_contour(roi):
    contours, _ = cv2.findContours(roi.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)>0:
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt)>500:
            return cnt

def preProcessing(frame, count_frame):
    global hand, val
    val=[]
    global start
    roi = frame[70:300, 0:250]
    # img = frame[70:480, 10:640]
    cv2.rectangle(frame, (0, 70), (250, 300), (255, 0, 0), 1)

    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi = cv2.threshold(roi, 128, 255, cv2.THRESH_BINARY_INV)
    roi = cv2.medianBlur(roi, 3)

    cnt_max = get_contour(roi)
    if cnt_max is not None: # Return None if no contour is present
        # cv2.drawContours(frame, [cnt_max+(0,70)], -1, (0,255,0),2)
        if count_frame %30 ==0 or start:
            start = True
        if start: val = digit_pos(0)
        hand = getTop_point(cnt_max)

    cv2.imshow("Roi", roi)

    return hand, val



while True:
    digits_array = []
    _,frame=cap.read()
    frame=cv2.flip(frame,1)
    count_frame+=1

    hand, digits_array = preProcessing(frame,count_frame)
    total_digit = show_num(hand, digits_array)
    if len(hand)>0 and (hand[0] != -1 or hand[1] != -1) and count_frame % 30 == 0:
        letter = letter+ str(total_digit)
    else: letter=""+letter
    cv2.putText(frame, letter, (10, 40), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 2)

    cv2.imshow("Image", frame)
    key = cv2.waitKey(10)
    if key & 0xFF == ord('q'):
        break