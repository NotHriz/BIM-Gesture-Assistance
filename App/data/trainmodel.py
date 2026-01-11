import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the CSV data
data = pd.read_csv('malay_sign_lang_coords.csv', header=None)

# Separate features and labels
X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # The last column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, max_depth=20,random_state=42)
print("Training the model...")
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f"Model accuracy: {accuracy * 100:.2f}%")
 
# Save the trained model
joblib.dump(model, 'App/models/msl_gesture_rf.joblib')

# Save the labels list
classes = model.classes_
joblib.dump(classes, 'App/models/classes.joblib')

print("Model and classes saved successfully.")