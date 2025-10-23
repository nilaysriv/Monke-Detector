# Monke-Detector
This Python program uses your computer's camera to detect how many fingers you're holding up. Based on the number of fingers, it displays a specific image on the screen.
It uses OpenCV for the camera feed and MediaPipe for hand tracking.

## Features
  * Detects hands and counts fingers (0-10).
  * Shows the live camera feed with hand landmarks (dots).
  * Displays a corresponding image for the number of fingers detected.
  * Includes a camera selection screen if you have multiple cameras.
  * Has a simple "anti-flicker" buffer to make the gesture detection more stable.

## Requirements
You need to have Python (<3.12) installed. Then, you can install the required libraries using pip:

```bash
pip install opencv-python mediapipe numpy
```

## How to Run

1.  **Run the Script:**
    Open a terminal or command prompt, navigate to the folder, and run:

    ```bash
    python monke.py
    ```
2.  **Select Camera:**
      * The program will first ask you to select a camera.
      * Press the number key (e.g., `0` or `1`) corresponding to the camera you want to use.

3.  **Initialization:**
      * Follow the on-screen instructions to show your right and left hands (palm and back) to the camera. This helps the detector get ready.

4.  **Use:**
      * Once initialized, show different finger counts (0-10) to the camera. The image in the 'Gesture Image' window will change.
      * Press **'q'** to quit the program.

## Demo

https://github.com/user-attachments/assets/b5e55d75-0d0b-4f8e-8deb-12135ef0650b

