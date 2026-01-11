import cv2
import mediapipe as mp
import os
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Setup the Hand Landmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE, # Use IMAGE mode for static files
    num_hands=2
)

# Initialize the detector
detector = vision.HandLandmarker.create_from_options(options)

DATASET_PATH = 'App\data\mydatabase' # Update this to your folder
output_data = []

# 2. Loop through each folder (Word)
for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(label_path): continue
    
    print(f"Processing word: {label}")
    
    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        image = cv2.imread(img_path)
        if image is None: continue

        # Convert to MediaPipe format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 3. Extract Landmarks
        detection_result = detector.detect(mp_image)

        if detection_result.hand_landmarks:
            # We only need the first hand detected
            hand = detection_result.hand_landmarks[0]
            coords = []
            for lm in hand:
                coords.extend([lm.x, lm.y]) # Adds x and y (42 total)
            
            # Add the label (the word) to the end
            coords.append(label)
            output_data.append(coords)

# 4. Save to CSV
df = pd.DataFrame(output_data)
df.to_csv('malay_sign_lang_coords.csv', index=False, header=False)
print(f"Finished! Saved {len(df)} samples to CSV.")