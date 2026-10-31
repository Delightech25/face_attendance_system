import cv2
import os
import sys

name = sys.argv[1]

path = f"dataset/{name}"
os.makedirs(path, exist_ok=True)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cam = cv2.VideoCapture(0)

count = 0

while True:

    ret, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:

        count += 1

        cv2.imwrite(f"{path}/{count}.jpg", gray[y:y+h,x:x+w])

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow("Capturing Faces", img)

    if cv2.waitKey(1)==27 or count>=30:
        break

cam.release()
cv2.destroyAllWindows()