import cv2
import mediapipe as mp
import numpy as np
import time

# --- 1. Initialize MediaPipe Hands and Drawing Utilities ---
# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,        # Set to False for video stream processing
    max_num_hands=1,                # Detect only one hand
    min_detection_confidence=0.7,   # Minimum confidence for hand detection
    min_tracking_confidence=0.5     # Minimum confidence for hand tracking
)

# Initialize MediaPipe Drawing utilities for drawing landmarks
mp_drawing = mp.solutions.drawing_utils

# --- 2. Drawing Board Parameters ---
# Canvas dimensions (will be updated based on webcam feed)
canvas_width, canvas_height = 1280, 720

# Create a blank canvas (NumPy array) for drawing
# This canvas will hold the persistent drawing lines
# Initialize with white background (255, 255, 255 for BGR channels)
drawing_canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8) + 255

# Drawing state variables
is_drawing = False  # True when pinch gesture is detected
last_point = None   # Stores the last point (x, y) for drawing lines
drawing_color = (0, 0, 255)  # Default drawing color (BGR: Blue, Green, Red) -> Red
line_thickness = 5           # Default line thickness

# Colors for UI elements (BGR format)
CURSOR_COLOR_MOVING = (255, 255, 255) # White
CURSOR_COLOR_DRAWING = (0, 0, 255)    # Red
LANDMARK_COLOR = (0, 255, 0)         # Green
CONNECTION_COLOR = (255, 0, 0)       # Blue

# --- 3. Webcam Setup ---
cap = cv2.VideoCapture(0) # Open default webcam (index 0)

if not cap.isOpened():
    print("Error: Could not open webcam. Please ensure it's connected and not in use by other applications.")
    exit()

# Set webcam resolution (optional, helps ensure consistent canvas size)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, canvas_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, canvas_height)

# --- 4. Main Drawing Loop ---
print("Virtual Drawing Board started. Show your hand to the camera.")
print("Move your index finger to control the cursor. Pinch (thumb to index) to draw.")
print("Press 'c' to clear the canvas.")
print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Flip the frame horizontally for a mirror effect (common for webcam applications)
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    # Create a copy of the live frame to draw UI elements on
    display_frame = frame.copy()

    # Overlay the persistent drawing from drawing_canvas onto the current frame
    # We use a blend operation to combine the drawing with the live video
    # Only draw where drawing_canvas is not white (i.e., where lines are drawn)
    mask = drawing_canvas != 255 # Create a mask where drawing exists
    display_frame[mask] = drawing_canvas[mask] # Copy drawing pixels

    # Initialize current_point for drawing
    current_point = None

    if results.multi_hand_landmarks:
        # Assuming only one hand is detected (max_num_hands=1)
        hand_landmarks = results.multi_hand_landmarks[0]

        # Get the coordinates of the index finger tip (landmark 8)
        # and thumb tip (landmark 4)
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

        # Convert normalized coordinates to pixel coordinates
        cursor_x = int(index_tip.x * canvas_width)
        cursor_y = int(index_tip.y * canvas_height)
        current_point = (cursor_x, cursor_y)

        # Calculate Euclidean distance between index tip and thumb tip for pinch detection
        # Normalized coordinates are between 0 and 1, so scale by canvas width/height
        # A smaller distance indicates a pinch
        pinch_distance = np.sqrt(
            ((index_tip.x - thumb_tip.x) * canvas_width)**2 +
            ((index_tip.y - thumb_tip.y) * canvas_height)**2
        )

        # Define a threshold for pinch detection (adjust as needed)
        # This threshold is relative to the canvas width to make it somewhat scale-independent
        pinch_threshold = canvas_width * 0.05 # e.g., 5% of canvas width

        # Determine if currently drawing based on pinch gesture
        current_is_drawing = pinch_distance < pinch_threshold

        # --- Drawing Logic ---
        if current_is_drawing:
            # If we just started drawing (wasn't drawing before), set the starting point
            if not is_drawing:
                last_point = current_point
            is_drawing = True
            # Draw a line on the persistent drawing_canvas
            if last_point:
                cv2.line(drawing_canvas, last_point, current_point, drawing_color, line_thickness)
            last_point = current_point # Update last_point for the next frame
            cv2.putText(display_frame, "Drawing!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, CURSOR_COLOR_DRAWING, 2, cv2.LINE_AA)
        else:
            is_drawing = False
            last_point = None # Reset last point when not drawing
            cv2.putText(display_frame, "Moving", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, CURSOR_COLOR_MOVING, 2, cv2.LINE_AA)

        # Draw landmarks and connections on the display frame (for visual feedback)
        mp_drawing.draw_landmarks(
            display_frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=LANDMARK_COLOR, thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=CONNECTION_COLOR, thickness=2, circle_radius=2)
        )

        # Draw the cursor (index finger tip) on the display frame
        cursor_color = CURSOR_COLOR_DRAWING if is_drawing else CURSOR_COLOR_MOVING
        cv2.circle(display_frame, current_point, line_thickness + 5, cursor_color, -1) # Filled circle
        cv2.circle(display_frame, current_point, line_thickness + 5, (0,0,0), 2) # Black border

    else:
        # No hand detected
        is_drawing = False
        last_point = None
        cv2.putText(display_frame, "No Hand Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the combined frame
    cv2.imshow('Virtual Drawing Board', display_frame)

    # --- Keyboard Controls ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): # Press 'q' to quit
        break
    elif key == ord('c'): # Press 'c' to clear the drawing canvas
        drawing_canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8) + 255 # Reset to white
        print("Canvas cleared!")

# --- 5. Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("Virtual Drawing Board closed.")
