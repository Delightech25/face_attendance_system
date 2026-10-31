import cv2
import os
import pyttsx3
from datetime import datetime

engine = pyttsx3.init()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

names = os.listdir("dataset")

cam = cv2.VideoCapture(0)

attendance_marked = []

while True:

    ret,img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,1.2,5)

    for(x,y,w,h) in faces:

        id,confidence = recognizer.predict(gray[y:y+h,x:x+w])

        if confidence < 60:

            name = names[id]

            if name not in attendance_marked:

                attendance_marked.append(name)

                now = datetime.now()

                os.makedirs("attendance", exist_ok=True)

                with open("attendance/attendance.csv","a") as f:
                    f.write(f"\n{name},{now.strftime('%Y-%m-%d')},{now.strftime('%H:%M:%S')}")

                message = f"Welcome {name}. Attendance Marked"

                engine.say(message)
                engine.runAndWait()

        else:
            name = "Unknown"

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.putText(img,name,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,255,0),2)

        cv2.putText(img,"Attendance Marked",
                    (x,y+h+30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,(0,255,0),2)

    cv2.imshow("Classroom Attendance",img)

    if cv2.waitKey(1)==27:
        break

cam.release()
cv2.destroyAllWindows()