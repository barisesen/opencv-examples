#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import cv2
import imageio

face_cascade = cv2.CascadeClassifier('data/haarcascade-frontalface-default.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascade-eye.xml')

def detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return frame

image = imageio.imread('media/people.jpg')
image = detect(frame=image)
imageio.imwrite('outputs/people.jpg', image)
