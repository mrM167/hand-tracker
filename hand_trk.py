import cv2
import mediapipe as mp 
import time 

capt =cv2.VideoCapture(0)

mphand=mp.solutions.hands
hands=mphand.Hands(False)
mpDraw=mp.solutions.drawing_utils

pTime=0
cTime=0

while True:
    success, img= capt.read()
    

    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB )
    result = hands.process(imgRGB)

    if result.multi_hand_landmarks:
        for handlm in result.multi_hand_landmarks:
            for id, lm  in enumerate(handlm.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w), int(lm.y*h)
                print(id,cx,cy)
                if id==0:
                    cv2.circle(img, (cx,cy),10, (255,0,255),cv2.FILLED)
            mpDraw.draw_landmarks(img,handlm,mphand.HAND_CONNECTIONS)
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    cv2.imshow('IMAGE',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capt.release()
cv2.destroyAllWindows()
