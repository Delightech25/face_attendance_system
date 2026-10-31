# Face Attendance System

An automated computer vision and facial recognition-based attendance tracking system designed to streamline check-ins, eliminate proxy attendance, and maintain secure, real-time logs for educational institutions and corporate workplaces.

---

## 📖 Overview
Traditional attendance systems (such as manual roll calls, biometric fingerprint scanners, or RFID cards) are often time-consuming, prone to bottlenecks, or susceptible to buddy-punching (proxy attendance). The **Face Attendance System** leverages modern computer vision and deep learning techniques to detect and recognize faces from a live video stream or image capture, automatically recording accurate attendance timestamps into a secure database.

---

## 🔑 Key Features & Objectives
* **Real-Time Facial Recognition:** Identifies registered individuals instantly from live camera feeds or snapshots.
* **Automated Logging:** Automatically records attendance timestamps, user IDs, and status (Present/Absent) to a database or CSV file.
* **Anti-Proxy Protection:** Ensures high accuracy to prevent unauthorized check-ins and eliminate buddy-punching.
* **Administrative Dashboard / Export:** Simplifies record management with exportable attendance reports (CSV/Excel) for administrators and managers.

---

## 🛠️ Core Architecture & Pipeline

1. **Dataset Collection & Registration:** 
   * Captures and stores reference face images or facial embedding vectors for authorized users.
2. **Face Detection & Alignment:** 
   * Detects bounding boxes around faces in live frames using models like Haar Cascades, HOG, or Multi-Task Cascaded Convolutional Networks (MTCNN).
3. **Feature Extraction & Recognition:** 
   * Computes facial embeddings and matches them against registered profiles using distance metrics or classifiers (e.g., FaceNet, LBPH, or DeepFace).
4. **Attendance Logging & Reporting:** 
   * Marks attendance upon successful match, ensuring duplicate check-ins within the same time window are prevented, and logs the records.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Computer Vision & Image Processing:** OpenCV, Dlib / DeepFace
* **Machine Learning / Deep Learning:** Scikit-Learn, TensorFlow / PyTorch (depending on underlying face recognition modules)
* **Data Processing & Storage:** Pandas, NumPy, SQLite / CSV logs
* **User Interface (Optional):** Streamlit, Flask, or Tkinter

---

## ⚙️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Delightech25/face-attendance-system.git](https://github.com/Delightech25/face-attendance-system.git)
   cd face-attendance-system
