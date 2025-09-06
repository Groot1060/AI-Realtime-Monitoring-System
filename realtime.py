import cv2
import face_recognition
import pickle
import numpy as np
from datetime import datetime
import mysql.connector  # MySQL connection

# Database connection
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Vishal@2003",
    database="ai_monitoring"
)
cursor = conn.cursor()

# Load trained model
with open("face_recognition_model.pkl", 'rb') as f:
    clf, classes = pickle.load(f)

# Define periods (1-hour slots)
periods = {
    1: (10, 11), 2: (11, 12), 3: (12, 13), 4: (13, 14),
    5: (14, 15), 6: (17, 18)
}

def get_current_period():
    """Determine the current period based on the current hour"""
    current_hour = datetime.now().hour
    for period, (start, end) in periods.items():
        if start <= current_hour < end:
            return period
    return None  # If outside class hours

def mark_attendance(student_id):
    """Mark attendance in the SQL database if not already marked for this period"""
    today = datetime.now().strftime("%Y-%m-%d")
    current_period = get_current_period()
    
    if current_period is None:
        return  # Outside class hours, do nothing

    # Check if the student is already marked present in this period
    cursor.execute("SELECT * FROM attendance WHERE student_id=%s AND date=%s AND period=%s",
                   (student_id, today, current_period))
    result = cursor.fetchone()

    if not result:  # If attendance is not marked yet
        cursor.execute("INSERT INTO attendance (student_id, date, period, status) VALUES (%s, %s, %s, %s)",
                       (student_id, today, current_period, "Present"))
        conn.commit()
        print(f"Attendance marked for Student ID: {student_id} in Period {current_period}")

# Start video capture
video_capture = cv2.VideoCapture(0)  # Use 0 for webcam or RTSP URL for CCTV

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # Detect faces
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)
    
    # Predict faces
    for (top, right, bottom, left), face_enc in zip(face_locations, face_encodings):
        # Scale back up face locations
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        # Predict using SVM
        proba = clf.predict_proba([face_enc])[0]
        confidence = np.max(proba)
        if confidence > 0.8:  # Confidence threshold
            student_id = classes[np.argmax(proba)]  # Assuming class labels contain student IDs
            mark_attendance(student_id)
        else:
            student_id = "Unknown"
        
        # Draw rectangle and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, student_id, (left + 6, bottom - 6), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display frame
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()

# Close database connection
cursor.close()
conn.close()
