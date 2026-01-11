import os
import cv2
import pandas as pd
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- SETTINGS ---
DATASET_PATH = 'App/data/mydatabase'
OUTPUT_CSV = 'malay_sign_lang_coords.csv'
MODEL_ASSET_PATH = 'hand_landmarker.task'

# Initialize MediaPipe for the images
base_options = python.BaseOptions(model_asset_path=MODEL_ASSET_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2
)
detector = vision.HandLandmarker.create_from_options(options)

all_rows = []
words = sorted([f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))])

for word in words:
    word_folder = os.path.join(DATASET_PATH, word)
    count = 0
    
    for filename in os.listdir(word_folder):
        file_path = os.path.join(word_folder, filename)
        all_coords_84 = [0.0] * 84
        found_hand = False

        # CASE A: Processing teammate's .txt files
        if filename.endswith(".txt"):
            with open(file_path, 'r') as f:
                coords = [float(x) for x in f.read().strip().split(',')]
                if len(coords) == 84:
                    all_coords_84 = coords
                    found_hand = True

        # CASE B: Processing your original .jpg/.png images
        elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(file_path)
            if image is None: continue
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            res = detector.detect(mp_image)
            
            if res.hand_landmarks:
                found_hand = True
                for i, hand_lms in enumerate(res.hand_landmarks):
                    if i >= 2: break
                    offset = i * 42
                    for j, lm in enumerate(hand_lms):
                        all_coords_84[offset + (j * 2)] = lm.x
                        all_coords_84[offset + (j * 2) + 1] = lm.y

        if found_hand:
            all_rows.append(all_coords_84 + [word])
            count += 1

# Save final master file
df = pd.DataFrame(all_rows)
df.to_csv(OUTPUT_CSV, index=False, header=False)
print(f"🚀 Master Dataset Created: {len(df)} samples from images AND text files.")