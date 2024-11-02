import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2

# Define paths
base_dir =  'images_for_training'  # Use an environment variable for flexibility

# Check if the directory exists
if not os.path.exists(base_dir):
    raise FileNotFoundError(f"The specified directory {base_dir} does not exist. Please provide a valid path.")

# Data Preprocessing with Data Augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# Data Generators
try:
    train_generator = datagen.flow_from_directory(
        base_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        base_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
except Exception as e:
    raise RuntimeError(f"Error during data generation: {e}")

# Use MobileNetV2 as the base model for transfer learning
base_model = MobileNetV2(input_shape=(150, 150, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base model

# Build the model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # Adjust the number of output classes as needed
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Use a lower learning rate for fine-tuning
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with error handling
try:
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=40,  # Adjust this based on your needs
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True)  # Use `.keras` extension
        ]
    )
except Exception as e:
    raise RuntimeError(f"Error during model training: {e}")

# Save the trained model
model.save("iris_classification_model.keras")  # Use `.keras` extension
print("Model saved successfully!")
