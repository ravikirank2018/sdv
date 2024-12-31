# Import necessary libraries
import cv2
import speech_recognition as sr
from flask import Flask, render_template, Response
import pyttsx3
import dlib
from scipy.spatial import distance
import threading

# Initialize Flask app
app = Flask(__name__)

# -------------------------------
# Text-to-Speech Engine
# -------------------------------
tts_lock = threading.Lock()

def speak(message):
    """Thread-safe TTS function"""
    print(f"AI Speaking: {message}")
    with tts_lock:
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 150)
        tts_engine.setProperty('volume', 0.9)
        tts_engine.say(message)
        tts_engine.runAndWait()

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

# Threshold for EAR below which eyes are considered closed
EAR_THRESHOLD = 0.2

# Minimum consecutive frames with closed eyes to trigger the question
EYE_CLOSED_FRAME_THRESHOLD = 10

# -------------------------------
# Voice Analysis Functionality
# -------------------------------
def listen_for_reply():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("Listening for voice reply...")
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            print(f"Heard: {text}")
            return text.lower()
    except Exception as e:
        print(f"Voice analysis error: {e}")
        return None

# -------------------------------
# Video Stream Functionality
# -------------------------------
consecutive_closed_frames = 0
monitoring = True

def analyze_face(frame):
    global consecutive_closed_frames, monitoring

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    print(f"Faces detected: {len(faces)}")  # Debug: Number of faces detected

    for face in faces:
        landmarks = shape_predictor(gray, face)
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        print(f"Avg EAR: {avg_ear}, Threshold: {EAR_THRESHOLD}")  # Debug: EAR values

        # Check if eyes are closed
        if avg_ear < EAR_THRESHOLD:
            consecutive_closed_frames += 1
            print(f"Consecutive closed frames: {consecutive_closed_frames}")  # Debug
            if consecutive_closed_frames >= EYE_CLOSED_FRAME_THRESHOLD:
                if monitoring:
                    monitoring = False
                    speak("Are you sleeping?")
                    reply = listen_for_reply()
                    if reply and "yes" in reply:
                        speak("Okay, take a break.")
                    else:
                        speak("Continuing to monitor.")
        else:
            if consecutive_closed_frames > 0:
                print(f"Eyes reopened. Resetting count.")  # Debug
            consecutive_closed_frames = 0
            monitoring = True

    return frame

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = analyze_face(frame)

            # Encode the frame for streaming
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
if __name__ == "__main__":
    # Speak greeting on startup
    threading.Thread(target=speak, args=("Hi, welcome to the AI-powered monitoring system.",)).start()

    # Run Flask app
    app.run(debug=True)
