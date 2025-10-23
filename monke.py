import cv2
import numpy as np
import mediapipe as mp
import math

def detect_cameras():
    """Checks for available camera devices and returns their indices."""
    print("Detecting available cameras...")
    available_cameras = []
    #Check camera indices from 0 to 9
    for i in range(10):
        cap_test = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap_test.isOpened():
            available_cameras.append(i)
            print(f"  - Camera index {i} found.")
        cap_test.release()
    if not available_cameras:
        print("Error: No cameras found.")
    return available_cameras

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0, #0 for fastest performance
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_drawing = mp.solutions.drawing_utils

cap = None

display_size = (400, 400)

#Image Mapping
image_files = {
    0: "IMG_20251007_140811_036 (1).jpg",    #0 Fingers (Fist)
    1: "IMG_20251004_180533_859 (1).jpg",    #1 Finger
    2: "IMG_20251009_215647_206.jpg",      #2 Fingers
    3: "IMG_20251009_215635_216.jpg",      #3 Fingers
    4: "IMG_20251007_140812_508 (1).jpg",    #4 Fingers (Surprised)
    5: "IMG_20251012_235918_568.jpg",      #5 Fingers (Scheming)
    6: "IMG_20251004_180535_201 (1).jpg",    #6 Fingers (Thinking)
    7: "IMG_20251009_215635_216.jpg",      #7 Fingers (Winking/Thinking)
    8: "IMG_20251007_140812_508 (1).jpg",    #8 Fingers (Surprised)
    9: "IMG_20251009_215647_206.jpg",      #9 Fingers (Attitude)
    10: "IMG_20251012_235918_568.jpg",     #10 Fingers (Excellent!)
}


#--- Load and Resize Images ---
image_map = {}
for key, file in image_files.items():
    img = cv2.imread(file)
    if img is not None:
        image_map[key] = cv2.resize(img, display_size)
    else:
        print(f"Error: Failed to load image '{file}'")

if len(image_map) != len(image_files):
    print("\nPlease make sure all image files are in the same folder as the script.")
    exit()

default_image = image_map[0]

def count_fingers_most_robust(hand_landmarks, handedness):
    """
    Final, most robust finger counting logic. It uses a stricter check for the four
    fingers and a stable reference for the thumb.
    """
    finger_count = 0
    tip_ids = [4, 8, 12, 16, 20] #Thumb, Index, Middle, Ring, Pinky
    
    for id in range(1, 5):
        tip_y = hand_landmarks.landmark[tip_ids[id]].y
        pip_y = hand_landmarks.landmark[tip_ids[id] - 2].y #PIP joint
        if tip_y < pip_y: #Finger is up if tip is above PIP joint
            finger_count += 1
            
    thumb_tip_x = hand_landmarks.landmark[tip_ids[0]].x
    index_mcp_x = hand_landmarks.landmark[tip_ids[1] - 3].x
    
    if handedness == 'Left': #Physical RIGHT hand (due to flip)
        if thumb_tip_x < index_mcp_x:
            finger_count += 1
    else: #Physical LEFT hand
        if thumb_tip_x > index_mcp_x:
            finger_count += 1
            
    return finger_count

app_state = "SELECT_CAMERA"
available_cameras = detect_cameras()

#Initialization steps
init_steps = ["SHOW_RIGHT_PALM", "SHOW_RIGHT_BACK", "SHOW_LEFT_PALM", "SHOW_LEFT_BACK"]
init_step_index = 0
init_confirmations = 0
INIT_CONFIRMATION_THRESHOLD = 30 #Require 30 frames of stable detection


confirmed_gesture_id = 0
last_detected_gesture = 0
gesture_buffer_count = 0
GESTURE_CONFIRMATION_THRESHOLD = 5 #Frames to confirm a gesture

while True:
    if app_state == "SELECT_CAMERA":
        #Create a black screen for selection
        frame = np.zeros((display_size[1], display_size[0] * 2, 3), dtype=np.uint8)
        cv2.putText(frame, "Select Camera:", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cam_text = "Press number key for camera: " + ", ".join(map(str, available_cameras))
        cv2.putText(frame, cam_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Camera Feed', frame)
        key = cv2.waitKey(1) & 0xFF
        
        #Check if a number key was pressed
        if ord('0') <= key <= ord('9'):
            cam_index = int(chr(key))
            if cam_index in available_cameras:
                cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
                if cap.isOpened():
                    print(f"Camera {cam_index} selected. Proceeding to initialization.")
                    app_state = "INITIALIZING"
                else:
                    print(f"Error: Failed to open camera {cam_index}.")
        if key == ord('q'):
            break
        continue #Skip the rest of the loop

    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame from camera. Exiting...")
        break
    
    frame = cv2.flip(frame, 1) #Flip for mirror view
    
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #Convert to RGB for MediaPipe
    image_rgb.flags.writeable = False #Performance optimization
    results = hands.process(image_rgb) #Process the image
    image_rgb.flags.writeable = True
    
    if app_state == "INITIALIZING":
        current_step = init_steps[init_step_index]
        instruction_text = ""
        hand_found = False

        if current_step == "SHOW_RIGHT_PALM":
            instruction_text = "Show PALM of your RIGHT hand"
            if results.multi_handedness:
                for handedness_obj in results.multi_handedness:
                    if handedness_obj.classification[0].label == 'Left':
                        hand_found = True
                        break
        
        elif current_step == "SHOW_RIGHT_BACK":
            instruction_text = "Show BACK of your RIGHT hand"
            if results.multi_handedness:
                for handedness_obj in results.multi_handedness:
                    if handedness_obj.classification[0].label == 'Left':
                        hand_found = True
                        break

        elif current_step == "SHOW_LEFT_PALM":
            instruction_text = "Show PALM of your LEFT hand"
            if results.multi_handedness:
                for handedness_obj in results.multi_handedness:
                    if handedness_obj.classification[0].label == 'Right':
                        hand_found = True
                        break
        
        elif current_step == "SHOW_LEFT_BACK":
            instruction_text = "Show BACK of your LEFT hand"
            if results.multi_handedness:
                for handedness_obj in results.multi_handedness:
                    if handedness_obj.classification[0].label == 'Right':
                        hand_found = True
                        break

        if hand_found:
            init_confirmations += 1
        else:
            init_confirmations = 0

        if init_confirmations > INIT_CONFIRMATION_THRESHOLD:
            init_confirmations = 0
            init_step_index += 1
            if init_step_index >= len(init_steps):
                app_state = "RUNNING"
        
        #Draw initialization instructions
        cv2.putText(frame, instruction_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if app_state == "RUNNING":
            cv2.putText(frame, "Initialization Complete!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    elif app_state == "RUNNING":
        total_finger_count = 0
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness_obj in zip(results.multi_hand_landmarks, results.multi_handedness):
                handedness = handedness_obj.classification[0].label
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                total_finger_count += count_fingers_most_robust(hand_landmarks, handedness)

        #Gesture stability buffer (anti-flicker)
        if total_finger_count == last_detected_gesture:
            gesture_buffer_count += 1
        else:
            last_detected_gesture = total_finger_count
            gesture_buffer_count = 0

        if gesture_buffer_count > GESTURE_CONFIRMATION_THRESHOLD:
            confirmed_gesture_id = last_detected_gesture
        
        #Display finger count
        cv2.putText(frame, "Fingers: " + str(confirmed_gesture_id), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        #Get gesture image
        gesture_image = image_map.get(confirmed_gesture_id, default_image)
        cv2.imshow('Gesture Image', gesture_image)

    #--- Display ---
    display_frame = cv2.resize(frame, display_size)
    cv2.imshow('Camera Feed', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): #Quit on 'q'
        break

if cap:
    cap.release()
cv2.destroyAllWindows()