import cv2
import numpy as np
import random

# Reading all the required haar cascade files
face_cascade=cv2.CascadeClassifier('/home/arjun/Pictures/opencv/FACE_MASK_PROJECT/mask_face.xml')
eye_cascade=cv2.CascadeClassifier('/home/arjun/Pictures/opencv/FACE_MASK_PROJECT/eye_face.xml')
mouth_cascade=cv2.CascadeClassifier('/home/arjun/Pictures/opencv/FACE_MASK_PROJECT/mouth_face.xml')

# Read video
video=cv2.VideoCapture(0)

while True:

    success,img=video.read()

    # required user messages
    

    # Converting it into gray
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Face detection
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)

    if len(faces)==0 :
        cv2.putText(img,text="No face found..",org=(30,30),fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale=1,color=(255,255,255),thickness=2)
    elif len(faces)==1:
        cv2.putText(img,text="face found..",org=(30,30),fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale=1,color=(255,255,255),thickness=2)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)


            # Detect lips counters
            mouth_rects=mouth_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4,minSize=(30, 30))


            # Face detected but Lips not detected which means person is wearing mask
        if(len(mouth_rects) == 0):
            cv2.putText(img,text="Thank You for wearing MASK",org=(30,30),fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale=1,color=(255,255,255),thickness=2)
        else:
            for (mx, my, mw, mh) in mouth_rects:
                
                if(y < my < y + h):
                    # Face and Lips are detected but lips coordinates are within face cordinates which `means lips prediction is true and
                    # person is not waring mask
                    cv2.putText(img,text='Please wear MASK ',org=(30,30),fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale=1,color=(255,255,255),thickness=2)
                    break

    # Show frame with results
    cv2.imshow('Mask Detection', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release video
video.release()
cv2.destroyAllWindows()
    
        
             

        

