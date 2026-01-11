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
        """
        Run MediaPipe detection, produce features for the classifier, and return:
          - landmarks: list of hands, each hand is a list of (x_norm, y_norm) tuples
                       (normalized coordinates in [0..1], relative to image width/height)
                       Example: [ [(x1,y1),(x2,y2),...],  # hand 0
                                  [(x1,y1),...] ]       # hand 1 (if any)
                     If no hands detected returns [].
          - prediction: formatted string like "Label (NN.NN%)"
        
        Notes:
        - The classifier expects a flattened list [x1,y1,x2,y2,...] for a single hand;
          here we use the first detected hand for prediction to preserve existing behavior.
        - Returning normalized landmarks allows the GUI to draw skeletons at any resolution.
        """
        try:
            # Convert frame for MediaPipe (SRGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            # Detect landmarks
            result = self.detector.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            # If MediaPipe fails for any reason, return no landmarks and an error label
            print("MediaPipe detection error:", e)
            return [], f"Error: {e}"

        if not result.hand_landmarks:
            return [], "No Hand"

        # Collect landmarks for all detected hands (normalized coords)
        hands_landmarks = []
        flattened_coords_per_hand = []
        for hand_landmarks in result.hand_landmarks:
            coords_flat = []
            landmarks_norm = []
            for lm in hand_landmarks:
                coords_flat.extend([lm.x, lm.y])
                landmarks_norm.append((lm.x, lm.y))
            flattened_coords_per_hand.append(coords_flat)
            hands_landmarks.append(landmarks_norm)

        # Use the first detected hand for prediction (keeps compatibility with existing model)
        try:
            coords_for_model = flattened_coords_per_hand[0]
            prediction_label = self.model.predict([coords_for_model])[0]
            probs = self.model.predict_proba([coords_for_model])[0]
            confidence = max(probs)
            prediction_str = f"{prediction_label} ({confidence:.2%})"
        except Exception as e:
            # If classifier fails, still return landmarks so GUI can draw the skeleton
            print("Classification error:", e)
            prediction_str = f"Error: {e}"

        # Return list of hands' normalized landmarks and the prediction string
        return hands_landmarks, prediction_str