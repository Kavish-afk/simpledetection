import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
fullbody_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cap = cv2.VideoCapture(0)

# cap = cv2.VideoCapture('.mp4')

while True:

    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # for (x, y, w, h) in faces:
    # cv2.rectangle(img, (x, y), (x+w, y+h), (254, 0, 0), 2)
    # smile = smile_cascade.detectMultiScale(gray, 1.1, 2)
    # for (x, y, w, h) in smile:
    # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)
    # fullbody = fullbody_cascade.detectMultiScale(gray, 1.1, 2)
    # for(x, y, w, h) in fullbody:
    # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)
    eye = eye_cascade.detectMultiScale(gray, 1.1, 2)
    for (x, y, w, h) in eye:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
    cv2.imshow('img', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
