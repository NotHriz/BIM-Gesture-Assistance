import joblib
import mediapipe as mp
# ... other imports at the top!

class GestureEngine:
    def __init__(self):
        # 1. Load the "Brain"
        self.model = joblib.load('models/msl_gesture_rf.joblib')
        
        # 2. Load the "Dictionary" (The labels we just discussed!)
        self.labels = joblib.load('models/classes.joblib')
        
        # 3. Setup MediaPipe
        # (Add your HandLandmarker initialization here)

    def process_frame(self, frame, timestamp_ms):
        # ... (MediaPipe processing logic) ...
        
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
        
        return frame, "No Hand Detected"