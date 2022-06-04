import cv2
import time
import mediapipe as mp

cap=cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw=mp.solutions.drawing_utils


p_time=0

current_time=0

while True:
    _ , img = cap.read()

    img_RGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(img_RGB)

    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for HandLMS in results.multi_hand_landmarks:
            for id , lm in enumerate(HandLMS.landmark):
                #print(id,lm)
                h,w,c=img.shape
                cx,cy=int(lm.x * w),int(lm.y *h)
                print(id,cx,cy)
                if id==0:
                    cv2.circle(img,(cx,cy),10,(255,0,0),cv2.FILLED)
            mpDraw.draw_landmarks(img,HandLMS,mpHands.HAND_CONNECTIONS)
    current_time=time.time()
    Fps=1/(current_time-p_time)
    p_time=current_time
    #print(Fps)

    cv2.putText(img,str(int(Fps)),(20,70),cv2.FONT_HERSHEY_SIMPLEX,3,(255,100,45),2)
    cv2.imshow("image", img)
    cv2.waitKey(1)


