

import cv2
import time
import mediapipe as mp

class HandDetector() :
    def __init__(self,mode=False,max_Hands=2, complexity = 1,detection_cons=0.5,track_conf=0.5):
        self.mode=mode
        self.maxHands=max_Hands
        self.detConf=detection_cons
        self.trackConf=track_conf
        self.complex=complexity
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.complex, self.detConf, self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands (self,img , draw=True):
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_RGB)

        if self.results.multi_hand_landmarks:
            for self.HandLMS in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, self.HandLMS, self.mpHands.HAND_CONNECTIONS)
        return img
    def findPosition(self, img,handNo=0,draw=True):
        lmList=[]

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(my_hand.landmark):

                #print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)
                lmList.append([id,cx,cy])
                #if id == 0:
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        return lmList


    #print(results.multi_hand_landmarks)





def main():

    p_time = 0
    cap = cv2.VideoCapture(0)
    current_time = 0
    detector = HandDetector()
    while True:
        _, img = cap.read()
        img = detector.findHands(img=img)
        lmList= detector.findPosition(img)
        if len(lmList)!=0:
            print(lmList[4])
        current_time = time.time()
        fps = 1 / (current_time - p_time)
        p_time = current_time
        # print(Fps)

        cv2.putText(img, str(int(fps)), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 100, 45), 2)
        cv2.imshow("image", img)
        cv2.waitKey(1)





if __name__ == "__main__":
    main()