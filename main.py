import cv2
import mediapipe as mp
import time
import csv
import math
import os
from datetime import datetime
from boltiot import Bolt
from typing import List, Tuple, Optional

# Constants
EAR_THRESHOLD = 0.2
CONSECUTIVE_FRAMES_THRESHOLD = 7
FRAME_SKIP_RATE = 3
ALERT_SLEEP_TIME = 2
CSV_FILENAME = "eye_alert_log.csv"

# Eye landmark indices for MediaPipe
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Load credentials from environment variables for security
API_KEY = os.getenv("BOLT_API_KEY", "f9a2549d-8635-4e40-8f12-c9e1fc389731")
DEVICE_ID = os.getenv("BOLT_DEVICE_ID", "BOLT3848117")

def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def calculate_ear(eye_landmarks):
    A = euclidean_distance(eye_landmarks[1], eye_landmarks[5])
    B = euclidean_distance(eye_landmarks[2], eye_landmarks[4])
    C = euclidean_distance(eye_landmarks[0], eye_landmarks[3])
    return (A + B) / (2.0 * C)

def log_eye_closure(timestamp, duration, response_time=None):
    filename = "eye_alert_log.csv"
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["Timestamp", "Duration (seconds)", "Response Time (ms)"])
        writer.writerow([timestamp, duration, response_time])

def main():
    eye_blink_count = 0
    isClose = 0
    count = 0
    blink_flag = False
    buzz = 0
    eyes_closed_start_time = None

    EAR_THRESHOLD = 0.2
    CONSEC_FRAMES = 7

    cap = cv2.VideoCapture(0)
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()

            count += 1
            if count % 3 != 0:
                continue

            if not ret:
                print("Ignoring empty camera frame.")
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:

                    h, w, _ = frame.shape
                    left_eye_indices = [33, 160, 158, 133, 153, 144]
                    right_eye_indices = [362, 385, 387, 263, 373, 380]

                    left_eye = [(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in left_eye_indices]
                    right_eye = [(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in right_eye_indices]

                    left_ear = calculate_ear(left_eye)
                    right_ear = calculate_ear(right_eye)
                    avg_ear = (left_ear + right_ear) / 2.0



                    if avg_ear < EAR_THRESHOLD:
                        isClose += 1
                        if eyes_closed_start_time is None:
                            eyes_closed_start_time = time.time()  # start time of closure

                        if blink_flag == False:
                            eye_blink_count += 1
                            blink_flag = True
                        cv2.rectangle(frame, (0, 0), (100, 50), (0, 0, 255), -1)
                        cv2.putText(frame, 'closed', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    else:
                        isClose = 0
                        blink_flag = False
                        eyes_closed_start_time = None  # reset
                        cv2.rectangle(frame, (0, 0), (100, 50), (0, 255, 0), -1)
                        cv2.putText(frame, 'open', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.rectangle(frame, (0, 51), (700, 150), (4, 55, 35), -1)
            cv2.putText(frame, "Blinked : " + str(eye_blink_count), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (25, 225, 135), 4)

            if isClose > 7:
                buzz = 1
                cv2.rectangle(frame, (0, 252), (700, 150), (4, 55, 35), -1)
                cv2.putText(frame, "Drive Alert!! : Eyes are Closed for long time", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (25, 225, 135), 4)

                # 📝 Log alert
                if eyes_closed_start_time is not None:
                    
                    detection_to_alert_start = time.time()  # ⏱️ Start timer before sending API
                    duration = round(detection_to_alert_start - eyes_closed_start_time, 2)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # 🛰️ Send command to Bolt
                    response_on = mybolt.digitalWrite('0', 'HIGH')
                    buzzer_trigger_time = time.time()  # ⏱️ After API call
                    response_time = round((buzzer_trigger_time - detection_to_alert_start) * 1000, 2)  # in milliseconds

                    # 📋 Log response time (append to same CSV or print)
                    log_eye_closure(timestamp, duration, response_time)
                    print(f"[{timestamp}] Response Time: {response_time} ms | Duration: {duration}s | API Response: {response_on}")
                    


                    eyes_closed_start_time = None  # reset after logging
                    time.sleep(2)
                    mybolt.digitalWrite('0', 'LOW')
                    break

            else:
                buzz = 0

            cv2.imshow('Eyes Detection', frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()