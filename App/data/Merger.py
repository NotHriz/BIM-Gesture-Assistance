import os
import pandas as pd
import time

# --- SETTINGS ---
EXISTING_CSV = 'malay_sign_lang_coords.csv'
NEW_DATA_DIR = 'my_new_data'
OUTPUT_CSV = 'malay_sign_lang_coords.csv' # Overwriting the same file with combined data

start_time = time.time()
new_rows = []

# 1. Load the existing data (the 3,000 images you already processed)
if os.path.exists(EXISTING_CSV):
    print(f"📖 Loading existing dataset: {EXISTING_CSV}")
    df_existing = pd.read_csv(EXISTING_CSV, header=None)
    print(f"✅ Found {len(df_existing)} existing samples.")
else:
    print("⚠️ No existing CSV found. Starting a fresh one.")
    df_existing = pd.DataFrame()

# 2. Process ONLY the new .txt files from teammates
if os.path.exists(NEW_DATA_DIR):
    words = [f for f in os.listdir(NEW_DATA_DIR) if os.path.isdir(os.path.join(NEW_DATA_DIR, f))]
    
    for word in words:
        word_folder = os.path.join(NEW_DATA_DIR, word)
        files = [f for f in os.listdir(word_folder) if f.endswith('.txt')]
        print(f"📂 Extracting {len(files)} new samples for: {word}")
        
        for filename in files:
            file_path = os.path.join(word_folder, filename)
            try:
                with open(file_path, 'r') as f:
                    # Convert text "0.1, 0.2..." into a list of floats
                    coords = [float(x) for x in f.read().strip().split(',')]
                    if len(coords) == 84:
                        new_rows.append(coords + [word.lower()])
            except Exception as e:
                print(f"❌ Error reading {filename}: {e}")

# 3. Combine and Save
if new_rows:
    df_new = pd.DataFrame(new_rows)
    # Combine old and new
    df_final = pd.concat([df_existing, df_new], ignore_index=True)
    
    # Remove potential duplicates (in case you ran this twice)
    df_final = df_final.drop_duplicates()
    
    df_final.to_csv(OUTPUT_CSV, index=False, header=False)
    
    print("\n" + "="*30)
    print(f"🚀 MERGE COMPLETE!")
    print(f"📈 Old Samples: {len(df_existing)}")
    print(f"➕ New Samples Added: {len(new_rows)}")
    print(f"📊 Total Dataset Size: {len(df_final)}")
    print(f"⏱️ Time Taken: {time.time() - start_time:.2f} seconds")
    print("="*30)
else:
    print("ℹ️ No new text data found to append.")