import cv2
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
fullbody_cascade= cv2.CascadeClassifier('haarcascade_fullbody.xml')
img = cv2.imread('dp.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#faces = face_cascade.detectMultiScale(gray, 1.1, 4)
#for (x, y, w, h) in faces:
#    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
fullbody = fullbody_cascade.detectMultiScale(gray, 1.1, 4)
for (x, y, w, h) in fullbody:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.imshow('img', img)
cv2.waitKey()