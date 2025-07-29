import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.utils import to_categorical

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.5)  # Lowered for better detection

# Load full data
train_df = pd.read_csv('sign_mnist_train.csv')  # Full 27,455 samples
test_df = pd.read_csv('sign_mnist_test.csv')    # Full 7,172 samples

# Prepare data
def preprocess_image(row):
    img = row.values[1:].reshape(28, 28).astype('float32') / 255.0
    img_uint8 = cv2.resize((img * 255).astype(np.uint8), (128, 128))  # Upscale to 128x128
    img_rgb = cv2.cvtColor(cv2.merge([img_uint8, img_uint8, img_uint8]), cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    if result.multi_hand_landmarks:
        landmarks = []
        for hand_lms in result.multi_hand_landmarks:
            for lm in hand_lms.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            landmarks_array = np.array(landmarks)
            if landmarks_array.shape[0] > 0:  # Accept any number of landmarks, not just 21
                return landmarks_array.flatten()
    return None

def preprocess_image_batch(df):
    X = []
    for _, row in df.iterrows():
        landmarks = preprocess_image(row)
        if landmarks is not None:
            X.append(landmarks)
    return np.array(X)

X_train = preprocess_image_batch(train_df)
y_train = to_categorical(train_df['label'].values[:X_train.shape[0]])  # Match y_train to X_train length
X_test = preprocess_image_batch(test_df)
y_test = to_categorical(test_df['label'].values[:X_test.shape[0]])  # Match y_test to X_test length

# Map labels to gestures
GESTURES = ['hand_up', 'hand_down', 'hand_left', 'hand_right']
label_map = {0: 0, 1: 1, 2: 2, 3: 3}  # Map first 4 labels to gestures
y_train_mapped = np.array([label_map.get(l, l % 4) for l in np.argmax(y_train, axis=1)])
y_test_mapped = np.array([label_map.get(l, l % 4) for l in np.argmax(y_test, axis=1)])
y_train = to_categorical(y_train_mapped, num_classes=4)
y_test = to_categorical(y_test_mapped, num_classes=4)

# Cache results
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print("Data loaded and preprocessed:", X_train.shape, y_train.shape)