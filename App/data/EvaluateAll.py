import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# 1. Load the Models
print("Loading models...")
rf_model = joblib.load('App/models/msl_gesture_rf.joblib')
svm_model = joblib.load('App/models/msl_gesture_svm.joblib')
svm_scaler = joblib.load('App/models/svm_scaler.joblib')
cnn_model = tf.keras.models.load_model('App/models/msl_gesture_cnn.h5')

# 2. Get some test data (Assume you have a test CSV)
# For the CNN, we would use the validation_generator from your training script
# For RF/SVM, we use the test split from the CSV

results = {
    "Model": ["Random Forest", "SVM", "CNN"],
    "Input Type": ["Coordinates", "Coordinates", "Raw Image"],
    "Accuracy (%)": [92.5, 94.1, 88.4] # Replace with your actual scores
}

df_results = pd.DataFrame(results)
print("\n--- FINAL PROJECT COMPARISON ---")
print(df_results)