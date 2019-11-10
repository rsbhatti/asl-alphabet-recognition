import cv2
import numpy as np

cap=cv2.VideoCapture(0)
i=0
while(1):
    try:
        ret,frame=cap.read()
        
        roi=frame[100:300,100:300],
        
        frame=cv2.flip(frame,1)
        cv2.rectangle(frame,(450,120),(600,400),(0,255,0),0)
        
        x=frame[120:370,450:600]
        x=cv2.flip(x,1)
        hsv=cv2.cvtColor(x,cv2.COLOR_BGR2HSV)
        
        lower_skin=np.array([0,20,70],dtype=np.uint8)
        upper_skin=np.array([20,255,255],dtype=np.uint8)
        
        mask=cv2.inRange(hsv,lower_skin,upper_skin)
        
        mask=cv2.GaussianBlur(mask,(5,5),100)
        
        cv2.imshow('frame',frame)
        cv2.imshow('cropped',x)
        cv2.imshow('mask',mask)
        cv2.imwrite('Nothing'+str(i)+'.jpg',mask)
        i+=1
        k=cv2.waitKey(5) & 0xFF
        if k==27:
            break
    except Exception as e:
        print(e)
        cv2.destroyAllWindows()
        cap.release()    
        break
cv2.destroyAllWindows()
cap.release()    
    