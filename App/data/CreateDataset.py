import cv2
import mediapipe as mp
import os
import pandas as pd
import numpy as np  # Needed for noisy_coords
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Setup the Hand Landmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2 # Detect up to 2
)
detector = vision.HandLandmarker.create_from_options(options)

DATASET_PATH = r'App\data\mydatabase'
output_data = []

# Force Alphabetical Order
labels = sorted([f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))])

for label in labels:
    label_path = os.path.join(DATASET_PATH, label)
    print(f"Processing word: {label}")
    
    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        image = cv2.imread(img_path)
        if image is None: continue

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        detection_result = detector.detect(mp_image)

        if detection_result.hand_landmarks:
            # --- 2-HAND LOGIC ---
            # Create a blank list of 84 zeros
            all_coords = [0.0] * 84 
            
            for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                if hand_idx >= 2: break # Only take first 2 hands
                
                offset = hand_idx * 42
                for i, lm in enumerate(hand_landmarks):
                    all_coords[offset + (i * 2)] = lm.x
                    all_coords[offset + (i * 2) + 1] = lm.y
            
            # Save original
            output_data.append(all_coords + [label])

            # --- DATA AUGMENTATION (Inside the loop!) ---
            # 1. Noisy version
            noisy = [c + np.random.normal(0, 0.002) if c != 0 else 0 for c in all_coords]
            output_data.append(noisy + [label])

            # 2. Scaled version
            scaled = [c * 1.02 if c != 0 else 0 for c in all_coords]
            output_data.append(scaled + [label])

# 4. Save to CSV
df = pd.DataFrame(output_data)
df.to_csv('malay_sign_lang_coords.csv', index=False, header=False)
print(f"Finished! Saved {len(df)} rows to CSV.")