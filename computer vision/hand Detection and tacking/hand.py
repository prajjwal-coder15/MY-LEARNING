import cv2
import sys
import mediapipe 
import time

cap = cv2.VideoCapture(0)

mpHands = mediapipe.solutions.hands 
hands = mpHands.Hands()
mpDraw = mediapipe.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmark)
 
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id , lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                if id ==11:
                 cv2.circle(img, (cx, cy),10,(255, 0, 255), cv2.FILLED)




            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime


    cv2.putText(img, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    cv2.imshow("fake face",img)
    cv2.waitKey(1)