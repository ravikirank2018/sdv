import tkinter as tk
from tkinter import filedialog, Label, Button, Radiobutton, IntVar
import cv2
from mediapipe import solutions as mp_solutions
from PIL import Image, ImageTk
import time
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp_solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp_solutions.drawing_utils


class PoseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pose Detection - Speed & Direction")
        self.root.geometry("800x600")

        # UI Elements
        self.label = Label(self.root, text="Choose Input Source:", font=("Arial", 16))
        self.label.pack(pady=10)

        self.input_choice = IntVar()
        self.radio_camera = Radiobutton(self.root, text="Laptop Camera", variable=self.input_choice, value=1,
                                        font=("Arial", 14))
        self.radio_file = Radiobutton(self.root, text="Video File", variable=self.input_choice, value=2,
                                      font=("Arial", 14))
        self.radio_camera.pack()
        self.radio_file.pack()

        self.load_button = Button(self.root, text="Start", command=self.start_detection, font=("Arial", 14))
        self.load_button.pack(pady=10)

        self.canvas = tk.Canvas(self.root, width=640, height=480)
        self.canvas.pack()

        self.cap = None
        self.prev_keypoints = None
        self.prev_time = time.time()

    def start_detection(self):
        # Stop any previous video capture
        if self.cap:
            self.cap.release()

        # Choose input source
        if self.input_choice.get() == 1:  # Laptop Camera
            self.cap = cv2.VideoCapture(0)
        elif self.input_choice.get() == 2:  # Video File
            file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
            if file_path:
                self.cap = cv2.VideoCapture(file_path)
            else:
                return
        else:
            return

        self.show_preview()

    def calculate_speed_direction(self, prev_point, curr_point, time_diff):
        if prev_point and curr_point:
            dx = curr_point[0] - prev_point[0]
            dy = curr_point[1] - prev_point[1]
            speed = np.sqrt(dx ** 2 + dy ** 2) / time_diff
            direction_angle = np.arctan2(dy, dx) * (180 / np.pi)  # Convert to degrees
            return speed, direction_angle
        return 0, 0

    def get_direction_text(self, angle):
        if -45 <= angle <= 45:
            return "Right"
        elif 45 < angle <= 135:
            return "Upward"
        elif -135 <= angle < -45:
            return "Downward"
        else:
            return "Left"

    def show_preview(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)

        if result.pose_landmarks:
            keypoints = [(lm.x, lm.y) for lm in result.pose_landmarks.landmark]
            curr_time = time.time()
            time_diff = curr_time - self.prev_time

            # Calculate speed and direction for a keypoint (e.g., nose)
            if self.prev_keypoints:
                speed, direction_angle = self.calculate_speed_direction(self.prev_keypoints[0], keypoints[0], time_diff)
                direction_text = self.get_direction_text(direction_angle)

                # Create a black rectangle for the text background
                box_start = (20, 20)
                box_end = (450, 150)
                cv2.rectangle(frame, box_start, box_end, (0, 0, 0), -1)  # Black background

                # Display speed, direction angle, and direction text
                font_scale = 1.5
                color = (255, 255, 255)  # White text color
                thickness = 3

                cv2.putText(frame, f"Speed: {speed:.2f} px/s",
                            (30, 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
                cv2.putText(frame, f"Angle: {direction_angle:.2f} deg",
                            (30, 100), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
                cv2.putText(frame, f"Direction: {direction_text}",
                            (30, 140), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

            self.prev_keypoints = keypoints
            self.prev_time = curr_time

            # Draw pose landmarks
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Resize and display frame in the UI
        frame_resized = cv2.resize(frame, (640, 480))
        img = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.canvas.image = img  # Keep a reference to avoid garbage collection

        # Continue updating the preview
        self.root.after(10, self.show_preview)

    def on_close(self):
        if self.cap:
            self.cap.release()
        self.root.destroy()


# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = PoseApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

