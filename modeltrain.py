import os
import cv2
import numpy as np
import face_recognition
import pickle
from sklearn import svm
from sklearn.preprocessing import LabelEncoder

# Function to load and process dataset
def load_dataset(dataset_dir):
    face_encodings = []
    labels = []
    
    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                image = face_recognition.load_image_file(image_path)
                
                # Detect faces
                face_locations = face_recognition.face_locations(image)
                if len(face_locations) == 0:
                    continue  # Skip images without detected faces
                
                # Get face encoding
                face_enc = face_recognition.face_encodings(image, known_face_locations=face_locations)[0]
                
                face_encodings.append(face_enc)
                labels.append(person_name)
    
    return np.array(face_encodings), np.array(labels)

# Main training function
def train_model(dataset_path, model_save_path):
    # Load dataset
    print("Loading dataset...")
    X, y = load_dataset(dataset_path)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train SVM classifier
    print("Training classifier...")
    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(X, y_encoded)
    
    # Save model
    with open(model_save_path, 'wb') as f:
        pickle.dump((clf, le.classes_), f)
    
    print(f"Model saved to {model_save_path}")
    return clf, le

if __name__ == "__main__":
    DATASET_PATH = r"C:\Users\kisho\Downloads\P3\dataset\process data\process"
    MODEL_PATH = "face_recognition_model.pkl"
    train_model(DATASET_PATH, MODEL_PATH)