import streamlit as st
import cv2
import os
import playsound
import torch
import numpy as np
import tempfile
from ultralytics import YOLO
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import joblib
from twilio.rest import Client
import json
from deepface import DeepFace
from collections import deque


# Fall Detection Model -> Fall -> detect_fall_type() [Movement Analysis] -> Fall Type -> Alert

# Twilio configuration [Change the Twilio Parameters]
ACCOUNT_SID = 'Your_Twilo_access_keys'
AUTH_TOKEN = 'Twilo_auth_token'
TWILIO_PHONE_NUMBER = 'twilo_reg_no'

COMMON_PHONE_NUMBER = "twilo_register_common_number"
# Parameters
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.75
ALERT_SOUND = "1secondalert.mp3"
PROFILE_FILE = "profiles.json"
FRAME_SKIP = 5                     # Faster Processing
FACE_VERIFICATION_INTERVAL = 30
FALL_ALERT_SENT = False
# Define the admin credentials
USERNAME = "admin"
PASSWORD = "admin"

#  Normal Fall threshold < Confidence < Suspicious Fall threshold

# 0.75< Confidence < 0.85

# Suspicious Fall
SUSPICIOUS_FALL_THRESHOLD = 0.85   # Higher confidence threshold for suspicious falls

# IF Suspicious Fall is recognized as Normal Fall, Lower the threshold of Normal Fall
# If Normal Fall is recognized as Suspicious Fall, Increase the threshold of Suspicious Fall

NORMAL_FALL_THRESHOLD = 0.75       # Lower threshold for normal falls
MIN_POST_FALL_FRAMES = 30          # Number of frames to observe after fall
MOVEMENT_THRESHOLD = 10            # Threshold for sudden movement detection

def load_yolo_model():
    return YOLO("best.pt")


def load_resnet_classifier():
    resnet = load_model("resnet_feature_extractor.h5")
    clf = joblib.load("fall_detection_model.pkl")
    return resnet, clf

# Load profile data
def load_profiles():
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r") as file:
            return json.load(file)
    return []

def get_latest_profile():
    profiles = load_profiles()
    return profiles[-1] if profiles else None

def get_registered_details():
    profiles = load_profiles()
    if profiles:
        return profiles[-1]
    return None

def play_alert():
    try:
        playsound.playsound(ALERT_SOUND)
    except Exception as e:
        print(f"Error playing sound: {e}")
        
def save_profile(name, relative_name, age, phone, image):
    profile_data = {"name": name, "relative_name": relative_name, "age": age, "phone": phone, "image": image}
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r") as file:
            profiles = json.load(file)
    else:
        profiles = []
    profiles.append(profile_data)
    with open(PROFILE_FILE, "w") as file:
        json.dump(profiles, file, indent=4)

def get_registered_phone():
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r") as file:
            profiles = json.load(file)
        return profiles[-1]["phone"] if profiles else None
    return None

def get_registered_relative_name():
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r") as file:
            profiles = json.load(file)
        return profiles[-1]["relative_name"] if profiles else None
    return None

def get_registered_name():
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r") as file:
            profiles = json.load(file)
        return profiles[-1]["name"] if profiles else None
    return None

def get_registered_age():
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r") as file:
            profiles = json.load(file)
        return profiles[-1]["age"] if profiles else None
    return None

def send_alert(person_name=None, relative_name=None, phone_number=None, message=None):
    if not phone_number:
        phone_number = COMMON_PHONE_NUMBER
        person_name = person_name or "Unknown Person"
        relative_name = relative_name or "Admin"
    
    if not message:
        message = f"Person {person_name} Fall Detected! Alerting {relative_name}."
    
    if not phone_number.startswith('+'):
        phone_number = f"+91{phone_number}"
    
    client = Client(ACCOUNT_SID, AUTH_TOKEN)
    try:
        message = client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=phone_number
        )
        st.warning(f"Alert sent to {relative_name}: {message}")
    except Exception as e:
        st.error(f"Failed to send alert: {e}")

# Optimized face recognition with caching

def get_face_embeddings():
    profiles = load_profiles()
    embeddings = []
    for profile in profiles:
        try:
            # Extract embedding vector properly
            result = DeepFace.represent(profile["image"], model_name='Facenet', enforce_detection=False)
            if result:
                embedding = result[0]['embedding']  # Get first face's embedding vector
                embeddings.append({
                    "name": profile["name"],
                    "relative_name": profile["relative_name"],
                    "phone": profile["phone"],
                    "embedding": embedding
                })
        except Exception as e:
            print(f"Error processing {profile['name']}'s image: {e}")
    return embeddings

def recognize_face_optimized(frame, embeddings):
    try:
        # Get face embedding from current frame
        result = DeepFace.represent(frame, model_name='Facenet', enforce_detection=False)
        if not result:
            return None
            
        current_embedding = result[0]['embedding']  # Get first face's embedding vector
        
        # Compare with all registered embeddings
        for profile in embeddings:
            # Convert to numpy arrays for vector operations
            registered_embedding = np.array(profile["embedding"])
            current_embedding_arr = np.array(current_embedding)
            
            # Calculate cosine similarity
            similarity = np.dot(registered_embedding, current_embedding_arr) / (
                np.linalg.norm(registered_embedding) * np.linalg.norm(current_embedding_arr)
            )  # Added missing closing parenthesis here
            
            if similarity > 0.7:
                return profile
    except Exception as e:
        print(f"Face recognition error: {e}")
    return None

# Optimized fall detection with frame skipping
def detect_fall_optimized(frame, model, frame_count, embeddings):
    global FALL_ALERT_SENT
    
    # Only process every FRAME_SKIP-th frame
    if frame_count % FRAME_SKIP != 0:
        return False
    
    # Initialize frame history for movement analysis
    if 'frame_history' not in st.session_state:
        st.session_state.frame_history = deque(maxlen=10)
        st.session_state.prev_gray = None
    
    results = model(frame, verbose=False)
    fall_detected = False
    suspicious_fall = False

    # Initialize face verification and fall tracking
    if 'face_verified' not in st.session_state:
        st.session_state.face_verified = None
        st.session_state.last_face_verification = -FACE_VERIFICATION_INTERVAL
        st.session_state.fall_start_frame = None
        st.session_state.post_fall_frames = 0
        st.session_state.movement_before_fall = False

    # Store current frame for movement analysis
    st.session_state.frame_history.append(frame.copy())
    
    # Face verification
    if (frame_count - st.session_state.last_face_verification) >= FACE_VERIFICATION_INTERVAL:
        matched_profile = recognize_face_optimized(frame, embeddings)
        if matched_profile:
            st.session_state.face_verified = matched_profile
        st.session_state.last_face_verification = frame_count

    # Movement analysis
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if st.session_state.prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(
            st.session_state.prev_gray, current_gray, 
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        if flow is not None:
            magnitude = np.sqrt(flow[...,0]*2 + flow[...,1]*2)
            if np.mean(magnitude) > MOVEMENT_THRESHOLD:
                st.session_state.movement_before_fall = True
    st.session_state.prev_gray = current_gray

    for result in results:
        for box in result.boxes:
            confidence = box.conf[0].item()
            label = result.names[int(box.cls[0])]
            
            if "Fall" in label:
                # Determine if fall is suspicious
                if confidence > SUSPICIOUS_FALL_THRESHOLD or st.session_state.movement_before_fall:
                    suspicious_fall = True
                    fall_type = "Suspicious Fall"
                    color = (0, 0, 255)  # Red
                    alert_type = "URGENT"
                elif confidence > NORMAL_FALL_THRESHOLD:
                    suspicious_fall = False
                    fall_type = "Normal Fall"
                    color = (0, 165, 255)  # Orange
                    alert_type = "Warning"
                    
                
                else:
                    continue
                
                # st.write(f"Fall Type: {fall_type}, Confidence: {confidence}")

                # Draw bounding box and label
                x_center, y_center, width, height = box.xywh[0]
                x1, y1 = int(x_center - width/2), int(y_center - height/2)
                x2, y2 = int(x_center + width/2), int(y_center + height/2)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{fall_type} {confidence:.2f}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, color, 2)
                

                if not FALL_ALERT_SENT:
                    play_alert()
                    if st.session_state.face_verified:
                        person_name = st.session_state.face_verified["name"]
                        relative_name = st.session_state.face_verified["relative_name"]
                        phone_number = st.session_state.face_verified["phone"]
                        
                        # Customize message based on fall type
                        if suspicious_fall:
                            message = f"🚨 {alert_type}: Suspicious fall detected for {person_name}! Immediate attention required! 🚨"
                        else:
                            message = f"⚠ {alert_type}: Normal Fall detected for {person_name}. Please check on them."
                            
                        send_alert(person_name, relative_name, phone_number, message)
                    else:
                        if suspicious_fall:
                            message = "🚨 URGENT: Suspicious fall detected! Unknown person may need help! 🚨"
                        else:
                            message = "⚠ Warning: Fall detected for unknown person. Please check the area."
                        send_alert("Unknown Person", "Admin", COMMON_PHONE_NUMBER, message)
                    
                    FALL_ALERT_SENT = True
                    fall_detected = True
                    st.session_state.fall_start_frame = frame_count

    # Track post-fall behavior
    if FALL_ALERT_SENT:
        st.session_state.post_fall_frames += 1
        if st.session_state.post_fall_frames > MIN_POST_FALL_FRAMES:
            FALL_ALERT_SENT = False
            st.session_state.post_fall_frames = 0
            st.session_state.fall_start_frame = None
            st.session_state.movement_before_fall = False

    return fall_detected

# Streamlit App
st.title("Fall Detection System")

# Login form
def login():
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.success("Login Successful!")
        else:
            st.error("Invalid credentials")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    FALL_ALERT_SENT = False

if not st.session_state.logged_in:
    login()
else:
    menu = ["Register Profile", "Fall Detection"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Register Profile":
        st.header("Register Your Profile")
        name = st.text_input("Name")
        relative_name = st.text_input("Relative's Name")
        age = st.number_input("Age", min_value=1, max_value=120, step=1)
        phone = st.text_input("Relative's Phone Number")

        photo_option = st.radio("Choose Photo Upload Method", ["Upload a Photo", "Take Screenshot with Camera"])

        if photo_option == "Upload a Photo":
            uploaded_photo = st.file_uploader("Upload a Photo", type=["jpg", "jpeg", "png"])
            if uploaded_photo:
                image_path = f"photos/{name}.jpg"
                os.makedirs("photos", exist_ok=True)
                image = Image.open(uploaded_photo)
                image.save(image_path)
        elif photo_option == "Take Screenshot with Camera":
            camera_photo = st.camera_input("Take a Photo")
            if camera_photo:
                image_path = f"photos/{name}.jpg"
                os.makedirs("photos", exist_ok=True)
                image = Image.open(camera_photo)
                image.save(image_path)
                
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Register"):
                if name and relative_name and age and phone and image_path:
                    save_profile(name, relative_name, age, phone, image_path)
                    st.success("Profile Registered Successfully!")
                else:
                    st.error("Please fill all fields and upload a photo or take a screenshot.")
        with col2:
            try:
                if st.button("Clear Data"):
                    os.remove(PROFILE_FILE)
                    st.success("Profile data cleared successfully!")
            except FileNotFoundError:
                st.warning("No profile data to clear.")

    elif choice == "Fall Detection":
        st.header("Fall Detection System")
        model = load_yolo_model()
        embeddings = get_face_embeddings()
        option = st.radio("Choose an option:", ("Video Upload", "Live Input Feed"))

        if option == "Video Upload":
            if 'face_verified' in st.session_state:
                del st.session_state.face_verified
            uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
            if uploaded_file:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                cap = cv2.VideoCapture(tfile.name)
                stframe = st.empty()
                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    detect_fall_optimized(frame, model, frame_count, embeddings)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    stframe.image(frame_rgb, caption="Live Fall Detection", use_container_width=True)
                
                cap.release()
                st.success("Video processing completed!")
                FALL_ALERT_SENT = False  # Reset alert flag

        elif option == "Live Input Feed":
            if 'face_verified' in st.session_state:
                del st.session_state.face_verified
            
            st.header("Live Fall Detection")
            start_button = st.button("Start Live Feed")
            stop_button = st.button("Stop Live Feed")
            
            if start_button:
                cap = cv2.VideoCapture(0)
                stframe = st.empty()
                frame_count = 0
                stop_processing = False
                
                while not stop_processing:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    detect_fall_optimized(frame, model, frame_count, embeddings)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    stframe.image(frame_rgb, caption="Live Fall Detection", use_container_width=True)
                    
                    # Check if stop button was pressed
                    if stop_button:
                        stop_processing = True
                
                cap.release()
                st.success("Live Feed Stopped!")
                FALL_ALERT_SENT = False  # Reset alert flag