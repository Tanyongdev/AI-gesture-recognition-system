import cv2 
import mediapipe as mp
import numpy as np
import pyautogui
import time
import math

def get_angle(a, b, c):
    """คำนวณมุมระหว่างจุดสามจุด a-b-c"""
    """คำนวณมุม 3D ระหว่างจุด a-b-c"""
    ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = math.degrees(math.acos(np.clip(cosine, -1.0, 1.0)))
    return angle


def count_fingers(lst):
    cnt = 0

    thresh = (lst.landmark[0].y*100 - lst.landmark[9].y*100)/2

    if (lst.landmark[5].y*100 - lst.landmark[8].y*100) > thresh:
        cnt += 1

    if (lst.landmark[9].y*100 - lst.landmark[12].y*100) > thresh:
        cnt += 1

    if (lst.landmark[13].y*100 - lst.landmark[16].y*100) > thresh:
        cnt += 1

    if (lst.landmark[17].y*100 - lst.landmark[20].y*100) > thresh:
        cnt += 1

    if (lst.landmark[5].x*100 - lst.landmark[4].x*100) > 6:
        cnt += 1

    return cnt 

cap = cv2.VideoCapture(0)

drawing = mp.solutions.drawing_utils    
hands = mp.solutions.hands
hand_obj = hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)


start_init = False 
prev = None

tap_interval = 0.5        # เวลาระหว่างการ tap แต่ละครั้ง
last_tap_time = 0         # เวลาที่ tap ล่าสุด
alt_holding = False       # ตอนนี้ Alt ถูกกดค้างอยู่ไหม

while True:
    end_time = time.time()
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)

    res = hand_obj.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    if res.multi_handedness:
        handedness = res.multi_handedness[0].classification[0].label
    
    if res.multi_hand_landmarks:

        hand_keyPoints = res.multi_hand_landmarks[0]

        if handedness == "Left":
            thumb_condition = hand_keyPoints.landmark[4].x > hand_keyPoints.landmark[3].x
        else:
            thumb_condition = hand_keyPoints.landmark[4].x < hand_keyPoints.landmark[3].x

        angles = [
            get_angle(hand_keyPoints.landmark[2], hand_keyPoints.landmark[3], hand_keyPoints.landmark[4]),   # thumb
            get_angle(hand_keyPoints.landmark[5], hand_keyPoints.landmark[6], hand_keyPoints.landmark[8]),   # index
            get_angle(hand_keyPoints.landmark[9], hand_keyPoints.landmark[10], hand_keyPoints.landmark[12]), # middle
            get_angle(hand_keyPoints.landmark[13], hand_keyPoints.landmark[14], hand_keyPoints.landmark[16]),# ring
            get_angle(hand_keyPoints.landmark[17], hand_keyPoints.landmark[18], hand_keyPoints.landmark[20]) # pinky
        ]

        finger_states = [1 if angle < 155 else 0 for angle in angles]

        cnt = count_fingers(hand_keyPoints)


        
        if not(prev == finger_states):
            if not(start_init):
                start_time = time.time()
                start_init = True

            elif (end_time-start_time) > 0.2:
                if (finger_states == [1, 0, 1, 1, 1]):
                    pyautogui.press("right") 

                elif (finger_states == [1, 0, 0, 1, 1]):
                    pyautogui.press("left")

                elif (finger_states == [0, 1, 1, 1, 1]):
                    pyautogui.press("up")

                elif (finger_states == [1, 1, 1, 1, 1]):
                    pyautogui.press("down")

                elif (finger_states == [0, 0, 0, 0, 0]):
                    pyautogui.press("space")

                elif (finger_states == [0, 0, 0, 1, 1]):
                     #  Alt + Tab
                  if not alt_holding:
                    # 👉 ครั้งแรกที่เข้า gesture                             
                    pyautogui.keyDown("alt")     # กด Alt ค้าง
                    pyautogui.press("tab")       # Tab ครั้งแรก           
                    alt_holding = True
                    last_tap_time = end_time

                elif  (finger_states == [0, 0, 1, 1, 1] and alt_holding):
                    if end_time - last_tap_time > tap_interval:
                        pyautogui.press("tab")     # กด Tab
                        last_tap_time = end_time  
            # 👉 ออกจาก gesture → ปล่อย Alt
            if finger_states != [0, 0, 0, 1, 1] and finger_states != [0, 0, 1, 1, 1]:
                if alt_holding:
                    pyautogui.keyUp("alt")
                    alt_holding = False
                    start_init = False
        else:
                prev = finger_states
                start_init = False
                print(finger_states)


        


        drawing.draw_landmarks(frm, hand_keyPoints, hands.HAND_CONNECTIONS)   

    cv2.imshow("window", frm)

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break