import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
import time

class GestureEngine:
    def __init__(self):
        # 1. Load the "Brain" (Expects 84 features)
        self.model = joblib.load('App/models/msl_gesture_rf.joblib')
        
        # 2. Load the "Dictionary"
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
        """
        Processes frame for 2-hand MSL recognition.
        Returns:
            - hands_landmarks: For GUI drawing
            - prediction_str: The translated word
        """
        try:
            # Convert frame for MediaPipe
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            # Detect landmarks
            result = self.detector.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            print("MediaPipe detection error:", e)
            return [], f"Error: {e}"

        if not result.hand_landmarks:
            return [], "No Hand Detected"

        # --- COORDINATE PROCESSING ---
        hands_landmarks_for_gui = []
        # Pre-fill 84 zeros (42 for Hand 1 + 42 for Hand 2)
        all_coords_84 = [0.0] * 84 

        # Loop through detected hands (MediaPipe returns up to 2)
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            if i >= 2: break  # Safety cap
            
            offset = i * 42
            current_hand_gui = []
            
            for j, lm in enumerate(hand_landmarks):
                # 1. Fill the 84-feature list for the Random Forest
                all_coords_84[offset + (j * 2)] = lm.x
                all_coords_84[offset + (j * 2) + 1] = lm.y
                
                # 2. Fill the list for the GUI skeleton drawing
                current_hand_gui.append((lm.x, lm.y))
                
            hands_landmarks_for_gui.append(current_hand_gui)

        # --- CLASSIFICATION ---
        try:
            # Predict using the padded 84-feature list
            prediction_label = self.model.predict([all_coords_84])[0]
            probs = self.model.predict_proba([all_coords_84])[0]
            confidence = max(probs)

            # Threshold: If AI is guessing with less than 60% confidence, don't show it
            if confidence < 0.60:
                prediction_str = "Analyzing..."
            else:
                prediction_str = f"{prediction_label} ({confidence:.2%})"
                
        except Exception as e:
            print("Classification error:", e)
            prediction_str = "Model Error"

        return hands_landmarks_for_gui, prediction_str