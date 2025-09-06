# AI Realtime Monitoring System 🎥🤖

An intelligent, real-time surveillance and attendance system using AI. This project integrates **CCTV live streaming**, **face recognition-based attendance**, and **violence detection** with an intuitive Flask-based web interface.

---

## 🚀 Features

- 🎯 **Face Recognition Attendance**  
  Detect and recognize registered faces using an SVM-based face recognition model. Automatically mark attendance at the start and end of class periods.

- 🎥 **Live CCTV Streaming**  
  Stream video in real-time using OpenCV and display it via Flask for monitoring.

- 🔍 **Violence Detection (YOLOv8)**  
  Detect violent activities using Ultralytics YOLOv8 model and save the incident as video clips.

- 📁 **User Management**  
  Store user registration data in MySQL and save media (photos/videos) locally using ERP ID-based filenames.

- 📊 **Time-Based Attendance**  
  Automatically determine current periods and mark attendance accordingly.

---

## 🧠 Tech Stack

| Component      | Technology                    |
|----------------|-------------------------------|
| Backend        | Python, Flask                 |
| AI Models      | Face Recognition (SVM), YOLOv8|
| Database       | MySQL                         |
| Frontend       | HTML, CSS (via Flask templates) |
| Camera         | RTSP/USB via OpenCV           |

---

## 📂 Project Structure

├── app.py # Main Flask application
├── shared_frame.py # Shared video stream handler
├── face_recognition_model.pkl# Trained SVM face recognition model
├── userdata/ # Stores user images & captured videos
├── detected_clips/ # Saved video clips from violence detection
├── templates/ # HTML templates
├── static/ # CSS, JS, etc.
├── requirements.txt
└── README.md



---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/rootvishal/AI-Realtime-Monitoring-System.git
cd ai-realtime-monitoring
2. Create Virtual Environment (Recommended)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. MySQL Setup
Create a MySQL database and user table.

Update your app.py with correct DB credentials.

sql
Copy
Edit
CREATE DATABASE surveillance_system;
USE surveillance_system;

CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(255),
    erp_id VARCHAR(100),
    ...
);
🚦 Running the App
bash
Copy
Edit
python app.py
Visit: http://127.0.0.1:5000/ in your browser.

📌 Future Enhancements
📧 Email/SMS alerts on violence detection

📱 Mobile interface

📊 Admin dashboard with stats

🔁 Multi-camera support



🤝 Contributing
Contributions, issues, and feature requests are welcome. Feel free to fork the repo and submit a pull request!

📬 Contact
Developer: Vishal Jadhav
Email: v707jadhav@gmail.com
LinkedIn: https://www.linkedin.com/in/vishal-jadhav-aiml/