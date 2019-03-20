import cv2 as cv
import numpy as np
face_cascade = cv.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')
smile_cascade = cv.CascadeClassifier('data/haarcascades/haarcascade_smile.xml')
leftEye = cv.imread("data/images/lefteye.png", cv.IMREAD_COLOR)
rightEye = cv.imread("data/images/righteye.png", cv.IMREAD_COLOR)
cv.namedWindow("Nayte Chandler")
cap = cv.VideoCapture(0)

while True:
    status, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        #print("test = ",y )
        #cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0))
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        #smile = smile_cascade.detectMultiScale(roi_gray)  
        eyes = eye_cascade.detectMultiScale(roi_gray)
        i = 0
        for (ex,ey,ew,eh) in eyes:
    
            if ey+eh < y and ew > w/5.5 and i < 2: #above middle of face
                #cv.rectangle(roi_color,(ex,eh),(ex+ew,ey+eh),(0,255,0),2)
                scaledW = int(ew*.9)
                scaledH = int(eh *.9)
                if ex < w/2:#left side of face

                    resized = cv.resize(leftEye, (scaledW, scaledH))
                    i = i + 1
                    roi_color[ey:ey+scaledH, ex:ex+scaledW] = resized

                if ex > w/2: #right side of face
                    resized = cv.resize(rightEye, (scaledW, scaledH))
                    i = i + 1
                    roi_color[ey:ey+scaledH, ex:ex+scaledW] = resized
     
    cv.imshow("Nayte Chandler", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cv.destroyAllWindows()
cap.release()
