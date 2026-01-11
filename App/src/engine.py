import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
import tensorflow as tf # Added for CNN

class GestureEngine:
    def __init__(self):
        # 1. Load All Models
        self.rf_model = joblib.load('App/models/msl_gesture_rf.joblib')
        self.svm_model = joblib.load('App/models/msl_gesture_svm.joblib')
        self.svm_scaler = joblib.load('App/models/svm_scaler.joblib')
        self.cnn_model = tf.keras.models.load_model('App/models/msl_gesture_cnn.h5')
        
        # 2. Settings
        self.labels = joblib.load('App/models/classes.joblib')
        self.active_model = "Random Forest" # Default
        
        # 3. Setup MediaPipe
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def set_model(self, model_name):
        self.active_model = model_name

    def _predict_cnn(self, frame):
        # CNN needs raw image resized to 224x224
        img = cv2.resize(frame, (224, 224))
        img = img / 255.0  # Rescale
        img = np.expand_dims(img, axis=0)
        preds = self.cnn_model.predict(img, verbose=0)[0]
        class_idx = np.argmax(preds)
        confidence = preds[class_idx]
        return self.labels[class_idx], confidence

    def process_frame(self, frame, timestamp_ms):
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, 
                                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = self.detector.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            return [], f"MediaPipe Error"

        # Initialize defaults
        hands_landmarks_for_gui = []
        all_coords_84 = [0.0] * 84 

        # Process Landmarks
        if result.hand_landmarks:
            for i, hand_landmarks in enumerate(result.hand_landmarks):
                if i >= 2: break
                offset = i * 42
                current_hand_gui = []
                for j, lm in enumerate(hand_landmarks):
                    all_coords_84[offset + (j * 2)] = lm.x
                    all_coords_84[offset + (j * 2) + 1] = lm.y
                    current_hand_gui.append((lm.x, lm.y))
                hands_landmarks_for_gui.append(current_hand_gui)

        # Classification Logic
        try:
            if self.active_model == "CNN":
                # CNN uses the raw image frame
                label, conf = self._predict_cnn(frame)
            elif self.active_model == "SVM":
                # SVM uses scaled coordinates
                scaled = self.svm_scaler.transform([all_coords_84])
                label = self.svm_model.predict(scaled)[0]
                conf = max(self.svm_model.predict_proba(scaled)[0])
            else: # Random Forest
                label = self.rf_model.predict([all_coords_84])[0]
                conf = max(self.rf_model.predict_proba([all_coords_84])[0])

            prediction_str = f"[{self.active_model}] {label} ({conf:.2%})"
        except Exception as e:
            prediction_str = "Classification Error"

        return hands_landmarks_for_gui, prediction_str