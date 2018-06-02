#! /usr/bin/python

import cv2

face_cascade = cv2.CascadeClassifier('data/haarcascade-frontalface-default.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascade-eye.xml')

# Load the overlay image: glasses.png
imgGlasses = cv2.imread('media/1.png')
imgHat = cv2.imread('media/hat3.png')


video_capture = cv2.VideoCapture(0)



while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, 30)
    

    for (x,y,w,h) in faces:
        # yuzun ust kismi
        hat_frame = frame[0:y, x:x+w]


        hx, hy, _ = hat_frame.shape

        hat = cv2.resize(imgHat, (hy, hx))

        sonuc = cv2.add(hat_frame, hat)
        
        sonuc_gray = cv2.cvtColor(sonuc, cv2.COLOR_BGR2GRAY)
        hx, hy, _ = sonuc.shape
        
        for i in range(hx):
            for j in range(hy):
                if sonuc_gray[i][j] == 255:
                    sonuc[i][j] = [0,0,0]

        frame[0:y, x:x+w] = sonuc
        

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3 , 10, 10)

        glassesDisplay = False
        totalWdith = 0
        totalHeight = 0
        xStart = 0
        xFinal = 0
        yStart = 0

        
        for (ex, ey, ew, eh) in eyes:
            if ex > xStart:
                xFinal = ex + eh

            if xStart == 0:
                xStart = ex
            
            if yStart == 0:
                yStart = ey


            if ex < xStart:
                xStart = ex
            if ey < yStart:
                yStart = ey    


            totalWdith = totalWdith + ew
            if totalHeight < eh:
                totalHeight = eh

            x1 = xStart
            x2 = xFinal
            y1 = yStart
            y2 = totalHeight + y1

            glassesWidth = x2 - x1
            glassesHeight = y2 - y1
            

            if glassesDisplay == True:
                
                glassesDisplay = False

                glasses = cv2.resize(imgGlasses, (glassesWidth,glassesHeight))

                # Iki gozun genislik ve yuksekliginin icinde kalan alani yakalar.
                roi = roi_color[y1:y2, x1:x2]

                # kopyasini aliyorum, resmi duzeltmek icin kullanacagim
                roi_temp = roi

                # iki resmi birlestir (renkli)
                roi = cv2.add(roi,glasses)
                
                # birlestirilen resmi gri yap
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                # gri resmin boyutlarini al
                x, y = roi_gray.shape

                for i in range(x):
                    for j in range(y):
                        if roi_gray[i][j] == 255:
                            roi[i][j] = roi_temp[i][j]
                        else:
                            roi[i][j] = [0,0,0]
                
                roi_color[y1:y2, x1:x2] = roi

            glassesDisplay = True    
 
    cv2.imshow('Video', frame)
    # cv2.waitKey()   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()