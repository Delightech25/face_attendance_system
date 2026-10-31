import cv2
import os
import json
import numpy as np

DATASET_PATH = "dataset"

faces = []
labels = []
label_map = {}

students = sorted(os.listdir(DATASET_PATH))

for label_id, student in enumerate(students):
    student_path = os.path.join(DATASET_PATH, student)
    if not os.path.isdir(student_path):
        continue

    label_map[str(label_id)] = student

    for image_file in os.listdir(student_path):
        img_path = os.path.join(student_path, image_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            faces.append(img)
            labels.append(label_id)

if not faces:
    print("No face data found in dataset.")
    exit(1)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

os.makedirs("trainer", exist_ok=True)
recognizer.save("trainer/trainer.yml")

with open("trainer/label_map.json", "w") as f:
    json.dump(label_map, f)

print(f"Training complete. {len(set(labels))} students, {len(faces)} samples.")
print("Label map:", label_map)