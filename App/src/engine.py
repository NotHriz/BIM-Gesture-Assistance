import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class GestureEngine:
    def __init__(self, model_path='hand_landmarker.task'):
        # 1. Initialize MediaPipe Task
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # 2. PLACEHOLDER: Load your Random Forest model here
        # self.rf_model = joblib.load('models/gesture_rf.pkl')

    def process_frame(self, frame, timestamp_ms):
        # Convert frame for MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Detect landmarks
        result = self.detector.detect_for_video(mp_image, timestamp_ms)
        
        prediction = "No Hand"
        
        if result.hand_landmarks:
            prediction = "Hand Detected" # Replace this with your RF model prediction
            for landmarks in result.hand_landmarks:
                # Draw landmarks on the frame for visual feedback
                for lm in landmarks:
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        return frame, prediction