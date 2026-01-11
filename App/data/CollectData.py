import cv2
import mediapipe as mp
import os
import time
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
MODEL_PATH = 'hand_landmarker.task'
SAVE_DIR = 'my_new_data'
SAMPLES_TO_COLLECT = 100 

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)
detector = vision.HandLandmarker.create_from_options(options)

word = input("Enter the Malay word you are signing: ").lower().strip()
word_folder = os.path.join(SAVE_DIR, word)
os.makedirs(word_folder, exist_ok=True)

# 1. FIND STARTING INDEX (To prevent overwriting friend's data)
existing_files = [f for f in os.listdir(word_folder) if f.endswith('.txt')]
start_idx = len(existing_files)
print(f"📂 Folder contains {start_idx} existing samples. Starting from sample_{start_idx}.txt")

cap = cv2.VideoCapture(1) # Change to 0 if using built-in webcam

print(f"\n🚀 Target: {word.upper()} | Starts in 3s...")
time.sleep(3)

count = 0
while count < SAMPLES_TO_COLLECT:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    res = detector.detect_for_video(mp_image, int(time.time() * 1000))

    # --- MID-POINT PAUSE LOGIC ---
    if count == 50:
        cv2.putText(frame, "PAUSED: CHANGE POSITION", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        cv2.imshow('MSL Data Collection', frame)
        cv2.waitKey(3000) # 3-second pause
        count += 1 # Move past the pause trigger

    # --- COLLECTION LOGIC ---
    if res.hand_landmarks:
        all_coords = [0.0] * 84 
        for hand_idx, landmarks in enumerate(res.hand_landmarks):
            if hand_idx >= 2: break
            offset = hand_idx * 42
            for i, lm in enumerate(landmarks):
                all_coords[offset + (i * 2)] = lm.x
                all_coords[offset + (i * 2) + 1] = lm.y
                cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 3, (0, 255, 0), -1)

        # Save using start_idx to ensure uniqueness
        coords_str = ",".join([str(c) for c in all_coords])
        file_path = os.path.join(word_folder, f"sample_{start_idx + count}.txt")
        with open(file_path, "w") as f:
            f.write(coords_str)
        
        count += 1
        print(f"Progress: {count}/{SAMPLES_TO_COLLECT}", end="\r")

    else:
        # Show warning if hand is missing so user knows why it's not collecting
        cv2.putText(frame, "NO HAND DETECTED", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # --- UI UPDATE (ALWAYS RUNS) ---
    cv2.putText(frame, f"Collected: {count}/{SAMPLES_TO_COLLECT}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow('MSL Data Collection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"\n✅ Done! Added {count} samples to {word_folder}")
cap.release()
cv2.destroyAllWindows()