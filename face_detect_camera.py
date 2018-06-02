#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import cv2
import imageio

face_cascade = cv2.CascadeClassifier('data/haarcascade-frontalface-default.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascade-eye.xml')

smile_cascade = cv2.CascadeClassifier('data/haarcascade-smile.xml')

def detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, ((0,frame.shape[0] -25)),(270, frame.shape[0]), (255,255,255), -1)
        cv2.putText(frame, "Number of faces detected: " + str(faces.shape[0]), (0,frame.shape[0] -10), cv2.FONT_HERSHEY_TRIPLEX, 0.5,  (0,0,0), 1)
 
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # roi_gray = gray[y:y+h, x:x+w]
        # roi_color = frame[y:y+h, x:x+w]
        # # eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)        
        # # for (ex, ey, ew, eh) in eyes:
        # #     cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # smiles = smile_cascade.detectMultiScale(roi_gray,
        #     scaleFactor= 1.7,
        #     minNeighbors=40,
        #     minSize=(50, 50))     

        # for (ex, ey, ew, eh) in smiles:
        #     print("güldün :D")
        #     cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     cv2.putText(roi_color,'Gülmeee :D',(0,130), font, 1, (200,255,155))  

    return frame


video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    canvas = detect(frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()