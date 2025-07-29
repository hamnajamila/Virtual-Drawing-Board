import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split # Still useful for general ML, though not strictly for train/test split here

# --- 1. Setup and Environment ---
# print("Setting up environment...")
# print("Installing required packages (if not already installed or outdated)...")
# # For this approach, mediapipe and its dependencies are not strictly needed
# # but keeping them for now if you might switch back or use it for live inference later.
# !pip install --upgrade tensorflow numpy pandas scikit-learn -q
# # Ensure OpenCV is still installed for image manipulation
# !pip install --upgrade opencv-python -q

# print("\nInstallation complete. Please RESTART RUNTIME (Runtime -> Restart runtime) if you see ModuleNotFoundErrors afterwards.")
# print("Then re-run all cells.")

# try:
#     import cv2
#     import tensorflow
#     import numpy
#     import pandas
#     # No mediapipe import needed for this direct CNN approach
# except ImportError as e:
#     print(f"Critical Error: Required library not found after installation. Please restart runtime and try again. Error: {e}")
#     exit()

# print("Libraries checked.")

# --- 2. Load and Preprocess Data for CNN ---
print("Loading and preprocessing data for CNN...")

try:
    train_df = pd.read_csv('sign_mnist_train.csv')
    test_df = pd.read_csv('sign_mnist_test.csv')
except FileNotFoundError:
    print("Error: Dataset files not found. Please ensure 'sign_mnist_train.csv' and 'sign_mnist_test.csv' are uploaded to your Colab environment or mounted from Google Drive.")
    from google.colab import files
    print("\nAttempting to upload files now...")
    try:
        uploaded = files.upload()
        if 'sign_mnist_train.csv' not in uploaded or 'sign_mnist_test.csv' not in uploaded:
            print("Upload failed or incomplete. Please manually upload the files.")
            exit()
        train_df = pd.read_csv('sign_mnist_train.csv')
        test_df = pd.read_csv('sign_mnist_test.csv')
        print("Files uploaded and loaded successfully.")
    except Exception as upload_e:
        print(f"An error occurred during file upload: {upload_e}")
        print("Please upload the files manually or ensure they are accessible.")
        exit()

# Prepare image data for CNN
def preprocess_images_cnn(df, target_size=(64, 64)): # CNNs can work with smaller images
    X_images = []
    y_labels = []
    for index, row in df.iterrows():
        label = row['label']
        pixel_data = row.drop('label').values # Get pixel values
        img = pixel_data.reshape(28, 28).astype('float32') / 255.0 # Normalize pixels to 0-1

        # Resize the image if desired (optional for 28x28, but can help with features)
        # Using 64x64 for slightly more detail without being too large
        if target_size != (28,28):
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

        X_images.append(img)
        y_labels.append(label)
    # Reshape to (num_samples, height, width, channels)
    # Add a channel dimension for grayscale images (1 channel)
    return np.array(X_images).reshape(-1, target_size[0], target_size[1], 1), np.array(y_labels)


X_train_images, y_train_raw = preprocess_images_cnn(train_df, target_size=(64, 64)) # Or (28,28)
X_test_images, y_test_raw = preprocess_images_cnn(test_df, target_size=(64, 64)) # Or (28,28)


print(f"Original train data size: {len(train_df)}. Processed train image data size: {X_train_images.shape[0]}")
print(f"Original test data size: {len(test_df)}. Processed test image data size: {X_test_images.shape[0]}")

# Adjust ASL labels (same as before)
def adjust_asl_labels(labels):
    adjusted_labels = []
    for label in labels:
        if label >= 9: # 'J' (index 9) is skipped in this static dataset
            adjusted_labels.append(label - 1)
        else:
            adjusted_labels.append(label)
    return np.array(adjusted_labels)

y_train_adjusted = adjust_asl_labels(y_train_raw)
y_test_adjusted = adjust_asl_labels(y_test_raw)

num_classes = len(np.unique(np.concatenate((y_train_adjusted, y_test_adjusted))))
print(f"Number of classes (ASL letters): {num_classes}")

y_train_one_hot = to_categorical(y_train_adjusted, num_classes=num_classes)
y_test_one_hot = to_categorical(y_test_adjusted, num_classes=num_classes)

X_train = X_train_images
y_train = y_train_one_hot
X_test = X_test_images
y_test = y_test_one_hot

print("Data loaded and preprocessed for CNN:")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# --- 3. Build and Train CNN Model ---
print("Building and training CNN model...")

model = Sequential([
    # First Conv Block
    Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]), # Input shape (height, width, channels)
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Second Conv Block
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Third Conv Block (Optional, for deeper networks)
    # Conv2D(128, (3, 3), activation='relu'),
    # BatchNormalization(),
    # MaxPooling2D((2, 2)),
    # Dropout(0.25),

    Flatten(), # Flatten the 2D feature maps to 1D vector
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5), # Higher dropout for classification layers

    Dense(num_classes, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

early_stopping = EarlyStopping(patience=20, restore_best_weights=True, monitor='val_loss') # Increased patience
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.000001, verbose=1) # Increased patience

history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=100,
                    batch_size=64, # Larger batch size for CNNs
                    callbacks=[early_stopping, reduce_lr])

# --- 4. Evaluate and Plot Results ---
print("Evaluating model...")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# --- 5. Save Model ---
print("Saving model...")
model_save_path = 'hand_gesture_cnn_model.keras' # Renamed for clarity
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# --- 6. Prediction Example for CNN (Optional) ---
def predict_gesture_cnn(image_path, model, target_size=(64,64), num_classes=24):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None

    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img_normalized = img_resized.astype('float32') / 255.0
    img_input = np.expand_dims(img_normalized, axis=0) # Add batch dimension
    img_input = np.expand_dims(img_input, axis=-1) # Add channel dimension for grayscale

    predictions = model.predict(img_input, verbose=0)[0]
    predicted_class = np.argmax(predictions)

    original_predicted_label = predicted_class
    if predicted_class >= 9:
        original_predicted_label += 1

    asl_alphabet = "ABCDEFGHIKLMNOPQRSTUVWXY"
    if 0 <= original_predicted_label < len(asl_alphabet):
        predicted_letter = asl_alphabet[original_predicted_label]
    else:
        predicted_letter = "Unknown ASL Letter"

    return predicted_class, predicted_letter, predictions

# Example usage (you'd need a sample image file)
# try:
#     # Upload a sample image, e.g., 'A.png' (make sure it's a grayscale hand sign image)
#     # from google.colab import files
#     # uploaded_sample = files.upload()
#
#     sample_image_path = 'A.png' # Replace with the name of your uploaded sample image
#     if os.path.exists(sample_image_path):
#         predicted_class_idx, predicted_letter, probabilities = predict_gesture_cnn(sample_image_path, model)
#         print(f"\nPrediction for {sample_image_path}:")
#         print(f"Predicted Class Index (adjusted): {predicted_class_idx}")
#         print(f"Predicted ASL Letter: {predicted_letter}")
#         print(f"Probabilities: {probabilities}")
#     else:
#         print(f"\nSample image not found at {sample_image_path}. Please upload a sample image or provide the correct path.")
# except Exception as e:
#     print(f"An error occurred during prediction example: {e}")