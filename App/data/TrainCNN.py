import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import os

# --- SETTINGS ---
# Path to your 3,500 images
DATA_DIR = 'App\data\mydatabase' 
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 12 # Start small to test

# 1. Image Pre-processing & Augmentation
# This "creates" new data by slightly zooming or shifting your images in memory
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2 # 20% for testing accuracy
)

train_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='training'
)

val_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation'
)

# 2. Setup MobileNetV2 (The "Pre-trained" Brain)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False # Freeze the base layers

# 3. Add custom layers for 10 MSL words
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x) # Prevents overfitting
predictions = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 4. Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Train
print(f"Training CNN on {train_gen.num_classes} classes...")
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# 6. Save
os.makedirs('App/models', exist_ok=True)
model.save('App/models/msl_gesture_cnn.h5')
print("Done! Saved as App/models/msl_gesture_cnn.h5")