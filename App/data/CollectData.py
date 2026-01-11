import cv2
import mediapipe as mp
import os
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
MODEL_PATH = 'hand_landmarker.task'
SAVE_DIR = 'my_new_data'  # Teammates can send you this whole folder
SAMPLES_TO_COLLECT = 100   # How many samples per word

# 1. Setup MediaPipe
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)
detector = vision.HandLandmarker.create_from_options(options)

# 2. Get User Input
word = input("Enter the Malay word you are signing (e.g., Makan): ").lower().strip()
word_folder = os.path.join(SAVE_DIR, word)

if not os.path.exists(word_folder):
    os.makedirs(word_folder)

# 3. Start Camera
cap = cv2.VideoCapture(1)

print(f"\nTarget: {word.upper()}")
print("Get ready! Recording starts in 3 seconds...")
for i in range(3, 0, -1):
    print(f"{i}...")
    time.sleep(1)

print("🚀 RECORDING STARTED!")

count = 0
while count < SAMPLES_TO_COLLECT:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1) # Mirror for natural feel
    h, w, _ = frame.shape
    
    # MediaPipe Processing
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    res = detector.detect_for_video(mp_image, int(time.time() * 1000))

    if res.hand_landmarks:
        # 1. Prepare a list to hold ALL coordinates (start with zeros for 2 hands = 84 points)
        all_coords = [0.0] * 84 
        
        # 2. Loop through detected hands (up to 2)
        for hand_idx, landmarks in enumerate(res.hand_landmarks):
            if hand_idx >= 2: break # Safety check
            
            # Fill the correct slot in our 84-point list
            # Hand 0 fills index 0-41 | Hand 1 fills index 42-83
            start_offset = hand_idx * 42
            for i, lm in enumerate(landmarks):
                all_coords[start_offset + (i * 2)] = lm.x
                all_coords[start_offset + (i * 2) + 1] = lm.y

            # 3. Draw dots for EVERY hand detected
            for lm in landmarks:
                cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 3, (0, 255, 0), -1)

        # 4. Save the full 84-point string to the file
        coords_str = ",".join([str(c) for c in all_coords])
        file_path = os.path.join(word_folder, f"sample_{count}.txt")
        with open(file_path, "w") as f:
            f.write(coords_str)

print(f"\n✅ Finished! 100 samples saved in {word_folder}")
cap.release()
cv2.destroyAllWindows()