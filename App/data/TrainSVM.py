import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load Dataset
DATA_PATH = 'malay_sign_lang_coords.csv'
df = pd.read_csv(DATA_PATH, header=None)

# 2. Split Features (X) and Labels (y)
X = df.iloc[:, :-1].values  # All coordinate columns (84)
y = df.iloc[:, -1].values   # The last column (Word label)

# 3. SVM is sensitive to scale! 
# We must scale the coordinates so they are between a similar range.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split into Training and Testing (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Initialize and Train SVM
# 'rbf' kernel is great for complex hand shapes
print("Training SVM Model...")
model = SVC(kernel='rbf', probability=True, C=10, gamma='scale')
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
print("\n--- SVM PERFORMANCE ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred))

# 7. Save Model and Scaler
# We MUST save the scaler too, or the GUI won't know how to process new data
joblib.dump(model, 'App/models/msl_gesture_svm.joblib')
joblib.dump(scaler, 'App/models/svm_scaler.joblib')
print("\n✅ SVM Model and Scaler saved to App/models/")