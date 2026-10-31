from flask import Flask, render_template, request, Response, jsonify, send_file
import cv2
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import pyttsx3
import time
import threading

app = Flask(__name__)

DATASET_PATH = "dataset"
ATTENDANCE_FILE = "attendance/attendance.csv"
TRAINER_PATH = "trainer/trainer.yml"
LABEL_MAP_PATH = "trainer/label_map.json"

camera = None
camera_running = False
camera_lock = threading.Lock()

attendance_marked = set()
attendance_lock = threading.Lock()

recognition_message = "Waiting for face..."
message_lock = threading.Lock()

tts_engine = pyttsx3.init()
tts_lock = threading.Lock()

recognizer = cv2.face.LBPHFaceRecognizer_create()
model_trained = False

label_map = {}

def load_model():
    global model_trained, label_map
    if os.path.exists(TRAINER_PATH) and os.path.exists(LABEL_MAP_PATH):
        recognizer.read(TRAINER_PATH)
        with open(LABEL_MAP_PATH) as f:
            label_map = json.load(f)
        model_trained = True

load_model()

faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def speak(message):
    def _speak():
        with tts_lock:
            tts_engine.say(message)
            tts_engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()


def train_model():
    global model_trained, label_map

    if not os.path.exists(DATASET_PATH):
        return False

    students = sorted(os.listdir(DATASET_PATH))
    if not students:
        return False

    faces = []
    labels = []
    new_label_map = {}

    for label_id, student in enumerate(students):
        student_path = os.path.join(DATASET_PATH, student)
        if not os.path.isdir(student_path):
            continue

        new_label_map[str(label_id)] = student

        for image_file in os.listdir(student_path):
            img_path = os.path.join(student_path, image_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                labels.append(label_id)

    if not faces:
        return False

    recognizer.train(faces, np.array(labels))
    os.makedirs("trainer", exist_ok=True)
    recognizer.save(TRAINER_PATH)

    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(new_label_map, f)

    label_map = new_label_map
    model_trained = True
    return True


def generate_frames():
    global recognition_message

    while True:
        with camera_lock:
            running = camera_running
            cam = camera

        if not running or cam is None:
            time.sleep(0.1)
            continue

        success, frame = cam.read()
        if not success:
            time.sleep(0.05)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = faceCascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces_detected:
            name = "Unknown"
            color = (0, 0, 255)

            if model_trained:
                label_id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

                if confidence < 60:
                    name = label_map.get(str(label_id), "Unknown")
                    color = (0, 255, 0)

                    today = datetime.now().strftime("%Y-%m-%d")
                    key = f"{name}_{today}"

                    with attendance_lock:
                        already_marked = key in attendance_marked

                    if not already_marked:
                        with attendance_lock:
                            attendance_marked.add(key)

                        now = datetime.now()
                        os.makedirs("attendance", exist_ok=True)

                        file_exists = os.path.exists(ATTENDANCE_FILE)
                        with open(ATTENDANCE_FILE, "a") as f:
                            if file_exists and os.path.getsize(ATTENDANCE_FILE) > 0:
                                f.write("\n")
                            f.write(f"{name},{now.strftime('%Y-%m-%d')},{now.strftime('%H:%M:%S')}")

                        with message_lock:
                            recognition_message = f"Attendance Marked: {name}"

                        speak(f"Welcome {name}")
                    else:
                        with message_lock:
                            recognition_message = f"Already Marked: {name}"
                else:
                    with message_lock:
                        recognition_message = "Unknown Face Detected"
            else:
                with message_lock:
                    recognition_message = "Model not trained yet"

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    total_students = 0
    if os.path.exists(DATASET_PATH):
        total_students = sum(
            1 for d in os.listdir(DATASET_PATH)
            if os.path.isdir(os.path.join(DATASET_PATH, d))
        )

    today_count = 0
    today_records = []
    all_records = []

    if os.path.exists(ATTENDANCE_FILE) and os.path.getsize(ATTENDANCE_FILE) > 0:
        df = pd.read_csv(ATTENDANCE_FILE, names=["Name", "Date", "Time"])
        df = df.dropna()
        today = datetime.now().strftime("%Y-%m-%d")
        today_df = df[df["Date"] == today]
        today_count = len(today_df)
        today_records = today_df.to_dict(orient="records")
        all_records = df.to_dict(orient="records")

    absent_count = max(0, total_students - today_count)

    return render_template(
        "dashboard.html",
        total_students=total_students,
        today_count=today_count,
        absent_count=absent_count,
        today_records=today_records,
        model_trained=model_trained,
        current_date=datetime.now().strftime("%B %d, %Y")
    )


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/start_camera")
def start_camera():
    global camera, camera_running
    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(0)
        camera_running = True
    return ("", 204)


@app.route("/stop_camera")
def stop_camera():
    global camera, camera_running
    with camera_lock:
        camera_running = False
        if camera is not None:
            camera.release()
            camera = None
    return ("", 204)


@app.route("/status")
def status():
    with message_lock:
        msg = recognition_message
    return jsonify({"message": msg, "model_trained": model_trained})


@app.route("/register")
def register_page():
    return render_template("register.html")


@app.route("/capture_faces", methods=["POST"])
def capture_faces():
    global camera, camera_running

    data = request.json
    name = data.get("name", "").strip()
    student_id = data.get("student_id", "").strip()

    if not name or not student_id:
        return jsonify({"status": "error", "message": "Name and ID required"}), 400

    # Stop live camera first to free device
    with camera_lock:
        camera_running = False
        if camera is not None:
            camera.release()
            camera = None

    time.sleep(0.5)

    folder = f"{DATASET_PATH}/{student_id}_{name}"
    os.makedirs(folder, exist_ok=True)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        return jsonify({"status": "error", "message": "Camera not accessible"}), 500

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    count = 0
    attempts = 0
    max_attempts = 300

    while count < 30 and attempts < max_attempts:
        ret, img = cam.read()
        if not ret:
            attempts += 1
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite(f"{folder}/{count}.jpg", gray[y:y+h, x:x+w])

        attempts += 1
        time.sleep(0.05)

    cam.release()

    if count == 0:
        return jsonify({"status": "error", "message": "No face detected. Try again."}), 400

    success = train_model()

    if not success:
        return jsonify({"status": "error", "message": "Training failed"}), 500

    return jsonify({"status": "registered", "samples": count, "name": name})


@app.route("/attendance")
def attendance_page():
    records = []
    if os.path.exists(ATTENDANCE_FILE) and os.path.getsize(ATTENDANCE_FILE) > 0:
        df = pd.read_csv(ATTENDANCE_FILE, names=["Name", "Date", "Time"])
        df = df.dropna()
        records = df.to_dict(orient="records")
    return render_template("attendance.html", records=records)


@app.route("/analytics")
def analytics_page():
    labels = []
    values = []
    if os.path.exists(ATTENDANCE_FILE) and os.path.getsize(ATTENDANCE_FILE) > 0:
        df = pd.read_csv(ATTENDANCE_FILE, names=["Name", "Date", "Time"])
        df = df.dropna()
        counts = df.groupby("Date").size().reset_index(name="count")
        labels = counts["Date"].tolist()
        values = counts["count"].tolist()
    return render_template("analytics.html", labels=json.dumps(labels), values=json.dumps(values))


@app.route("/export")
def export_attendance():
    if not os.path.exists(ATTENDANCE_FILE):
        return "No attendance data", 404

    df = pd.read_csv(ATTENDANCE_FILE, names=["Name", "Date", "Time"])
    df = df.dropna()
    export_path = "attendance/attendance_export.xlsx"
    df.to_excel(export_path, index=False)
    return send_file(export_path, as_attachment=True,
                     download_name="attendance.xlsx")


@app.route("/students")
def students_page():
    students = []
    if os.path.exists(DATASET_PATH):
        for folder in sorted(os.listdir(DATASET_PATH)):
            full_path = os.path.join(DATASET_PATH, folder)
            if os.path.isdir(full_path):
                parts = folder.split("_", 1)
                sid = parts[0] if len(parts) > 1 else "N/A"
                sname = parts[1] if len(parts) > 1 else folder
                sample_count = len(os.listdir(full_path))
                students.append({"id": sid, "name": sname, "samples": sample_count})
    return render_template("students.html", students=students)


if __name__ == "__main__":
    app.run(debug=True)