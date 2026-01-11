import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib

class GestureEngine:
    def __init__(self):
        # 1. Load the "Brain"
        self.model = joblib.load('App/models/msl_gesture_rf.joblib')
        
        # 2. Load the "Dictionary" (The labels we just discussed!)
        self.labels = joblib.load('App/models/classes.joblib')
        
        # 3. Setup MediaPipe
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
      
        

    def process_frame(self, frame, timestamp_ms):
        # ... (MediaPipe processing logic) ...
            # Convert frame for MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Detect landmarks
        result = self.detector.detect_for_video(mp_image, timestamp_ms)
        
        prediction = "No Hand" 
        
        if result.hand_landmarks:
            # Extract 42 coordinates
            coords = []
            for lm in result.hand_landmarks[0]:
                coords.extend([lm.x, lm.y])
            
            # PREDICTION
            # Predict returns the label directly if you trained with strings!
            prediction = self.model.predict([coords])[0]
            
            # OPTIONAL: Get probability (How sure is the AI?)
            probs = self.model.predict_proba([coords])[0]
            confidence = max(probs)
            
            return frame, f"{prediction} ({confidence:.2%})"
        
        return frame, prediction