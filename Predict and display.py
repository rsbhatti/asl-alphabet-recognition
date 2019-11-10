# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 22:22:24 2019

@author: Rajvindra
"""

import cv2
import numpy as np
from keras.preprocessing import image
import os
import show
cap = cv2.VideoCapture(0)
    
from keras.models import load_model
classifier = load_model('classifier.h5') 


while(1):
        
    try:  #an error comes if it does not find anything in window as it cannot find contour of max area
          #therefore this try error statement
          
        ret, frame = cap.read()
        
        
        #define region of interest
        roi=frame[100:300, 100:300]
        kernel = np.ones((3,3),np.uint8)
        
        
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame,(450,120),(600,370),(0,255,0),0)    
        
       
        
        cropped=frame[120:370,450:600]
        cropped = cv2.flip(cropped, 1)
       
        
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        
    # define range of skin color in HSV
        lower_skin = np.array([0,20,70], dtype=np.uint8)
        upper_skin = np.array([20,255,255], dtype=np.uint8)
        
     #extract skin colour image 
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        
    #blur the image
        mask = cv2.GaussianBlur(mask,(5,5),100) 
        cv2.imwrite('temp.jpg',mask)
        #show the windows
    
        test=image.load_img('temp.jpg',target_size=(64,64,3))
        x=image.img_to_array(test)
        x=np.expand_dims(x,axis=0)

        test=np.vstack([x])
        classes=classifier.predict_classes(test)
        print(classes)
        res=show.result(int(classes))
        cv2.putText(frame,res,(100,370),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,0),2,cv2.LINE_AA)
        cv2.imshow('frame',frame)
        cv2.imshow('mask',cropped)
    except Exception as e:
        print(e)
        cv2.destroyAllWindows()
        cap.release()    
        break
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
cv2.destroyAllWindows()
cap.release()  
os.remove('temp.jpg') 

  

'''
def adjust_gamma(image,gamma=1.0):
    invGamma=1.0/gamma
    table=np.array([((i/255.0)**invGamma)*255
                    for i in np.arange(0,256)]).astype("uint8")
    
    return cv2.LUT(image,table)

gamma=0.5
adjusted=adjust_gamma(cv2.imread("A.jpg"),gamma=0.4)
cv2.imshow("adjusted",adjusted)
cv2.imwrite('B.jpg',adjusted)
cv2.waitKey(0)
cv2.destroyAllWindows()'''