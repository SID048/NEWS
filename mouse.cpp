import cv2  # OpenCV library for image processing
import mediapipe as mp  # Mediapipe library for face and landmark detection
import pyautogui  # PyAutoGUI library for controlling the mouse and keyboard

# Initialize the camera for capturing video frames
cam = cv2.VideoCapture(0)

# Initialize Mediapipe FaceMesh module for facial landmark detection
# `refine_landmarks=True` ensures high precision for landmarks like eyes, mouth, etc.
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Get the screen dimensions for mapping landmarks to screen coordinates
screen_w, screen_h = pyautogui.size()

# Display a message to guide the user on how to exit the program
print("Press 'q' to exit the program")

# Main loop for processing the video feed frame by frame
while True:
    # Capture a frame from the camera
    _, frame = cam.read()

    # Flip the frame horizontally for natural interaction, as if looking in a mirror
    frame = cv2.flip(frame, 1)

    # Convert the frame from BGR (OpenCV default) to RGB (required by Mediapipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the RGB frame to detect facial landmarks
    output = face_mesh.process(rgb_frame)

    # Extract detected facial landmarks, if any
    landmark_points = output.multi_face_landmarks

    # Get the dimensions of the captured frame
    frame_h, frame_w, _ = frame.shape

    if landmark_points:  # Check if landmarks are detected
        # Extract the first face's landmarks (if multiple faces are detected)
        landmarks = landmark_points[0].landmark

        # **Mouse Movement Control**
        # Use landmarks (474 to 478) corresponding to the iris to control mouse cursor
        for id, landmark in enumerate(landmarks[474:478]):
            # Calculate the position of the landmark in the video frame
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)

            # Draw a small circle around the landmark for visualization
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            # Use the second landmark (id == 1) to control the mouse
            if id == 1:
                # Map the landmark's position to the screen dimensions
                screen_x = screen_w * landmark.x
                screen_y = screen_h * landmark.y

                # Move the mouse pointer to the mapped position
                pyautogui.moveTo(screen_x, screen_y)

        # **Eye Blink Detection**
        # Use landmarks (145 = upper eyelid, 159 = lower eyelid) for detecting blinks
        left_eye = [landmarks[145], landmarks[159]]

        for landmark in left_eye:
            # Calculate the position of the eyelid landmarks in the frame
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)

            # Draw a small circle around the eye landmarks for visualization
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

        # Blink Detection Logic
        # Measure the vertical distance between the upper and lower eyelid landmarks
        if (left_eye[0].y - left_eye[1].y) < 0.004:  # Adjust threshold for sensitivity
            # If the distance is below the threshold, register it as a blink
            pyautogui.click()  # Simulate a mouse click
            pyautogui.sleep(1)  # Add a short delay to avoid multiple clicks

        # **Mouth Detection for Scrolling**
        # Use landmarks (13 = upper lip, 14 = lower lip) for mouth movement detection
        mouth_upper = landmarks[13]  # Upper lip landmark
        mouth_lower = landmarks[14]  # Lower lip landmark

        # Calculate the vertical distance between the upper and lower lips
        mouth_gap = mouth_lower.y - mouth_upper.y

        # If the gap exceeds a certain threshold, detect mouth opening
        if mouth_gap > 0.02:  # Threshold for mouth opening (scroll down)
            pyautogui.scroll(-50)  # Scroll down the page
        elif mouth_gap < 0.01:  # Detect mouth closing (scroll up)
            pyautogui.scroll(50)  # Scroll up the page

    # **Display the Frame**
    # Add a title to the video feed for context
    cv2.putText(frame, "Eye Controlled Mouse", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the video feed with the drawn overlays in a window
    cv2.imshow('Eye Controlled Mouse', frame)

    # **Exit Condition**
    # Check if the user presses the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# **Release Resources**
# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
