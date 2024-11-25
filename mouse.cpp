import cv2
import mediapipe as mp
import pyautogui

# Initialize camera and modules
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# Display message
print("Press 'q' to exit the program")

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)  # Mirror the image for natural interaction
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark

        # Detect and move mouse cursor using iris landmarks (points 474 to 478)
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)  # Highlight landmarks
            if id == 1:  # Control the mouse cursor
                screen_x = screen_w * landmark.x
                screen_y = screen_h * landmark.y
                pyautogui.moveTo(screen_x, screen_y)

        # Eye blink detection using eye landmarks (points 145 and 159 for left eye)
        left_eye = [landmarks[145], landmarks[159]]
        for landmark in left_eye:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)  # Highlight eye landmarks

        # Blink detection logic
        if (left_eye[0].y - left_eye[1].y) < 0.004:  # Threshold for detecting blink
            pyautogui.click()
            pyautogui.sleep(1)

        # Scroll functionality using mouth landmarks (points 13 and 14)
        mouth_upper = landmarks[13]
        mouth_lower = landmarks[14]
        mouth_gap = mouth_lower.y - mouth_upper.y
        if mouth_gap > 0.02:  # Threshold for detecting mouth opening
            pyautogui.scroll(-50)  # Scroll down
        elif mouth_gap < 0.01:  # Detect mouth closing
            pyautogui.scroll(50)  # Scroll up

    # Display the frame with overlays
    cv2.putText(frame, "Eye Controlled Mouse", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Eye Controlled Mouse', frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()
