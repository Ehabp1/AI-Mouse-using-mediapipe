import numpy as np
import cv2 as cv
import mediapipe as mp
import pyautogui
import math

screen_width, screen_height = pyautogui.size()
print("Screen Width:", screen_width)
print("Screen Height:", screen_height)
x3 = 0
y3 =0

def mode(arr):
    fingers = []
    if arr[8][2] < arr[6][2]:
        fingers.append(1)
    else:
        fingers.append(0)
    if arr[12][2] < arr[10][2]:
        fingers.append(1)
    else:
        fingers.append(0)
    return fingers

video = cv.VideoCapture(0)
video.set(3,640)
video.set(4,480)

media = mp.solutions.hands
hand = media.Hands(min_detection_confidence=0.5)
draw = mp.solutions.drawing_utils
while True:
    success, frame = video.read()
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    rgb_frame = cv.flip(rgb_frame,1)
    result = hand.process(rgb_frame)
    loc = []
    if result.multi_hand_landmarks:
        for det in result.multi_hand_landmarks:
            for id, lm in enumerate(det.landmark):
                h, w, c = rgb_frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                loc.append([id,cx,cy])

            draw.draw_landmarks(rgb_frame, det, media.HAND_CONNECTIONS)
        up = mode(loc)
        x1,y1 = loc[8][1],loc[8][2]
        x2, y2 = loc[12][1], loc[12][2]
        length = math.hypot((loc[8][1] - loc[12][1]), (loc[8][2] - loc[12][2]))
        cv.rectangle(rgb_frame,(100, 100),(640-100,480 - 100),(0,255,0),3)


        #moving mode
        if up[0] and up[1] and length <30:
            pyautogui.click(x3, y3)
            #print('moving')
        #clicking mode
        if up[0] and not(up[1]):
            x3 = np.interp(x1,(100,480-100),(0,1920))
            y3 = np.interp(y1, (100, 460-100), (0,1080))
            pyautogui.moveTo(x3, y3)
           # print('click')

    cv.imshow('img',rgb_frame)
    if cv.waitKey(25) & 0XFF == ord('q'):
        break