import cv2
import mediapipe as mp
import pyautogui

# Initialize camera and face mesh model for facial landmark detection
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Get screen dimensions for mapping landmarks to screen coordinates
screen_w, screen_h = pyautogui.size()

# Main loop for processing video frames
while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect facial landmarks using Mediapipe
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks

    if landmark_points:
        landmarks = landmark_points[0].landmark

        # Use eye landmarks (474-478) for controlling the mouse pointer
        for id, landmark in enumerate(landmarks[474:478]):
            if id == 1:
                screen_x = screen_w * landmark.x
                screen_y = screen_h * landmark.y
                pyautogui.moveTo(screen_x, screen_y)

        # Detect blink (eye closing) to simulate a mouse click
        left_eye = [landmarks[145], landmarks[159]]
        if (left_eye[0].y - left_eye[1].y) < 0.004:
            pyautogui.click()

    # Display the processed frame
    cv2.putText(frame, "Eye Controlled Mouse", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Eye Controlled Mouse', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()
