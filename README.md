# Virtual Drawing Board (Hand Gesture Controlled)

This project transforms your webcam into an interactive drawing canvas, allowing you to create digital art using intuitive hand gestures. Powered by **MediaPipe** for real-time hand tracking and **OpenCV** for video processing, it offers a unique and engaging way to interact with a virtual drawing board — no mouse or touchscreen needed!

---

## Features

- **Hand Gesture Control**: Use your hand to control a virtual cursor.
- **Pinch-to-Draw**: Start drawing by bringing your thumb and index finger together (pinch gesture).
- **Real-time Tracking**: Hand landmarks and cursor update live on the webcam feed.
- **Adjustable Drawing**: (Fixed color and thickness in current version – easily customizable).
- **Clear Canvas**: Press `c` to erase everything on the board.
- **Mirror Effect**: Webcam is flipped horizontally for natural movement.

---

## Technologies Used

| Technology | Description |
|------------|-------------|
| Python | Programming language |
| OpenCV (`cv2`) | Webcam access, image processing |
| MediaPipe | Real-time hand tracking |
| NumPy | Efficient image/numerical data operations |

---

## Setup and Installation

### Prerequisites

- Python 3.9, 3.10, or 3.11
- A working webcam
- OS: Windows / macOS / Linux  
> ⚠️ Python 3.13 is **not fully supported** by MediaPipe/TensorFlow yet.

---

### Installation Guide

#### 1. Clone the Repository

```bash
git clone <your-github-repo-url>
cd <your-project-folder>
```

Setup and Installation
----------------------

### 2. Create a Virtual Environment (Recommended)

It's highly recommended to use a virtual environment to manage project dependencies and avoid conflicts with other Python projects.

```bash
   python -m venv venv_drawing_board   `
```
### 3. Activate the Virtual Environment

On Windows (Command Prompt/PowerShell):

```bash
   .\venv_drawing_board\Scripts\activate   `
```
On macOS/Linux (Bash/Zsh):

```bash
   source venv_drawing_board/bin/activate   `
```
✅ Your terminal prompt should now show (venv\_drawing\_board) indicating the environment is active.

### 4. Install Dependencies

With your virtual environment activated, install the required libraries:

```bash
   pip install opencv-python mediapipe numpy   `
```
▶️ How to Run
-------------

Ensure your virtual environment is activated (as shown in the "Setup and Installation" section).

Navigate to your project directory in the terminal and run the script:
```bash
   python drawing_board.py   `
```
🎮 How to Use
-------------

Once the application starts, a new window titled "Virtual Drawing Board" will appear, displaying your live webcam feed.

### Cursor Control

*   Position your hand clearly in front of the webcam.
    
*   The tip of your index finger will act as the virtual cursor.
    
*   Move your hand to move the cursor around the screen.
    
*   When you are not drawing, the cursor will appear as a white circle.
    

### Drawing Mode (Pinch Gesture)

*   To start drawing, bring your thumb and index finger together in a pinch gesture.
    
*   The cursor will change to a red circle (or your chosen drawing\_color if modified in code), indicating you are in drawing mode.
    
*   As you move your hand while pinching, a line will be drawn on the canvas.
    

### Stop Drawing

*   Release the pinch gesture (separate your thumb and index finger).
    
*   The cursor will revert to white, and you can move your hand without drawing.
    

### Clear Canvas

*   Press the c key on your keyboard to clear all drawings from the board.
    

### Quit Application

*   Press the q key on your keyboard to close the application window.
    

Future Enhancements
----------------------

*   **Color Picker**: Implement keyboard controls or on-screen UI to change the drawing color in real-time.
    
*   **Thickness Adjustment**: Add controls to change the line thickness.
    
*   **Eraser Tool**: Implement a specific gesture or key to switch to an eraser mode.
    
*   **Shape Drawing**: Add gestures to draw basic shapes (circles, squares).
    
*   **Save Drawing**: Functionality to save the created drawing as an image file.
    
*   **Multi-Hand Support**: Extend to use two hands for more complex interactions (e.g., one hand for drawing, one for controls).
    

Screenshot:
![image](https://github.com/user-attachments/assets/752a1e7a-c28f-47c0-9a0b-e4269f4eefce)


-------------

 Make sure your image is saved at images/screenshot.png inside your project directory.

License
----------

This project is open-source and available under the MIT License.
