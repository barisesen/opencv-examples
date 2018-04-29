#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import cv2
import imageio


face_cascade = cv2.CascadeClassifier('data/haarcascade-frontalface-default.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascade-eye.xml')


def detect(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1.3 = resmi kucultme oranı
    # min yakalaması gereken kare sayısı
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        gray_face = gray_frame[y:y+h, x:x+w]
        color_face = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(color_face, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return frame


reader = imageio.get_reader('media/people.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('outputs/people.mp4', fps=fps) 


for i, frame in enumerate(reader):
    frame = detect(frame=frame)
    writer.append_data(frame)
    print(i)        