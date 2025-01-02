# Import necessary libraries
import cv2
import speech_recognition as sr
from flask import Flask, render_template, Response
import pyttsx3
import dlib
from scipy.spatial import distance
import time

# Handle GPIO imports for non-Raspberry Pi systems
try:
    import RPi.GPIO as GPIO
except (ImportError, RuntimeError):
    print("RPi.GPIO not available. Using mock GPIO for testing.")
    from unittest.mock import Mock
    GPIO = Mock()

# Initialize Flask app
app = Flask(__name__)

# Initialize Text-to-Speech engine
tts_engine = pyttsx3.init()

def speak(message):
    tts_engine.say(message)
    tts_engine.runAndWait()

# -------------------------------
# PERCLOS Parameters
# -------------------------------
PERCLOS_WINDOW = 30  # Number of frames in the rolling window
CLOSED_EYE_FRAMES = {}  # Dictionary to track whether eyes were closed for the last N frames for each face
PERCLOS_THRESHOLD = 0.7  # Threshold for drowsiness (70% of frames closed)
EYE_CLOSED_TIMERS = {}  # Track closed eye time for each face
CLOSED_THRESHOLD = 60  # 60 seconds to mark as red
EAR_THRESHOLD = 0.2  # EAR threshold for eye closure

# -------------------------------
# Eye Aspect Ratio (EAR) Functionality
# -------------------------------
def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Load Dlib's pre-trained face detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks .dat")

# -------------------------------
# Analyze Faces with Timers and Status
# -------------------------------
def analyze_faces(frame):
    global EYE_CLOSED_TIMERS, CLOSED_EYE_FRAMES

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    face_data = []  # Store face rectangles and their respective status

    for idx, face in enumerate(faces):
        landmarks = shape_predictor(gray, face)

        # Extract eyes
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

        # Calculate EAR
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # Determine if eyes are closed
        is_closed = avg_ear < EAR_THRESHOLD

        # Initialize data for each face
        if idx not in EYE_CLOSED_TIMERS:
            EYE_CLOSED_TIMERS[idx] = 0
            CLOSED_EYE_FRAMES[idx] = []

        # Update rolling window for PERCLOS
        CLOSED_EYE_FRAMES[idx].append(is_closed)
        if len(CLOSED_EYE_FRAMES[idx]) > PERCLOS_WINDOW:
            CLOSED_EYE_FRAMES[idx].pop(0)

        # Calculate PERCLOS
        perclos = sum(CLOSED_EYE_FRAMES[idx]) / len(CLOSED_EYE_FRAMES[idx])

        if is_closed:
            EYE_CLOSED_TIMERS[idx] += 1  # Increment timer if eyes are closed
        else:
            EYE_CLOSED_TIMERS[idx] = 0  # Reset timer if eyes are open

        # Determine face status
        face_status = "red" if EYE_CLOSED_TIMERS[idx] > CLOSED_THRESHOLD else "green"

        face_data.append({
            "rect": (face.left(), face.top(), face.right(), face.bottom()),
            "status": face_status,
            "tag": f"Face {idx + 1}",
            "perclos": perclos
        })

    return face_data

# -------------------------------
# Draw Face Tags and Status on Frame
# -------------------------------
def draw_face_data(frame, face_data):
    for face in face_data:
        x1, y1, x2, y2 = face["rect"]
        color = (0, 0, 255) if face["status"] == "red" else (0, 255, 0)  # Red for danger, Green for safe

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw tag and PERCLOS
        cv2.putText(frame, f"{face['tag']} - {face['perclos']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

# -------------------------------
# Video Stream Functionality
# -------------------------------
def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            face_data = analyze_faces(frame)  # Analyze faces
            frame = draw_face_data(frame, face_data)  # Draw face data

            # Encode the frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# -------------------------------
# Flask Routes
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# -------------------------------
# Main Integration Logic
# -------------------------------
def main():
    camera = cv2.VideoCapture(0)

    try:
        while True:
            # Read frame from camera
            ret, frame = camera.read()
            if not ret:
                print("Failed to grab frame from camera.")
                continue

            # Analyze and draw face data
            face_data = analyze_faces(frame)
            frame = draw_face_data(frame, face_data)

            # Display the frame (for testing purposes)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Program interrupted.")

    finally:
        camera.release()
        GPIO.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Run Flask app for UI
    app.run(debug=True)

# -------------------------------
# Required pip packages
# -------------------------------
# Create a requirements.txt file with the following content:
# opencv-python
# SpeechRecognition
# pyaudio
# flask
# RPi.GPIO
# pyttsx3
# dlib
# scipy
