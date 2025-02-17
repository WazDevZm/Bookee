import cv2
import numpy as np
import mediapipe as mp
import pygame
import time
from twilio.rest import Client
#Implementing the second atempt of the cmputer vision project based on computer vision ision land makrs
# Twilio Configuration (Replace with your actual Twilio credentials)
TWILIO_SID = "your_twilio_sid"
TWILIO_AUTH_TOKEN = "your_twilio_auth_token"
TWILIO_PHONE_NUMBER = "your_twilio_phone_number"
EMERGENCY_CONTACT = "recipient_phone_number"

# Initialize Pygame for sound alerts
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("./beep-125033.mp3")  # Load an alert sound

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Define Eye and Mouth Landmarks (Mediapipe indices)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 17, 87, 57, 61]

EYE_THRESHOLD = 0.28  # Adjusted EAR threshold for better accuracy
MAR_THRESHOLD = 0.6  # Adjusted MAR threshold for better accuracy
FRAME_COUNT = 0  # Counter for consecutive drowsy frames
ALARM_ON = False

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye_landmarks, landmarks):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_landmarks]
    ear = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * np.linalg.norm(p1 - p4))
    return ear

# Function to calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth_landmarks, landmarks):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in mouth_landmarks]
    mar = np.linalg.norm(p2 - p5) / np.linalg.norm(p1 - p4)
    return mar

# Function to send SMS alert using Twilio
def send_alert():
    client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
    message = client.messages.create(
        body="ALERT: Driver is drowsy! Please check on them.",
        from_=TWILIO_PHONE_NUMBER,
        to=EMERGENCY_CONTACT
    )
    print("SMS Alert Sent!")

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = np.array([[lm.x * frame.shape[1], lm.y * frame.shape[0]] for lm in face_landmarks.landmark])
            
            left_EAR = eye_aspect_ratio(LEFT_EYE, landmarks)
            right_EAR = eye_aspect_ratio(RIGHT_EYE, landmarks)
            avg_EAR = (left_EAR + right_EAR) / 2.0
            
            mar = mouth_aspect_ratio(MOUTH, landmarks)
            
            if avg_EAR < EYE_THRESHOLD:
                FRAME_COUNT += 1
                if FRAME_COUNT >= 15:
                    cv2.putText(frame, "DROWSINESS ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    if not ALARM_ON:
                        pygame.mixer.Sound.play(alert_sound)
                        send_alert()  # Send SMS alert
                        ALARM_ON = True
            else:
                FRAME_COUNT = 0
                ALARM_ON = False
                cv2.putText(frame, "Driver Awake", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            
            if mar > MAR_THRESHOLD:
                cv2.putText(frame, "Yawning Detected!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    
    cv2.imshow("Drowsy Driver Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
