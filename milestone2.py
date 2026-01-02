import cv2
import mediapipe as mp
import math
import time
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

OUTPUT_DIR = "screenshots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Check permissions.")

# MediaPipe Hands model
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        gesture_name = "No Hand Detected"
        distance = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),  # Blue dots
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)  # Green lines
                )

                thumb = hand_landmarks.landmark[4]
                index = hand_landmarks.landmark[8]

                x1, y1 = int(thumb.x * w), int(thumb.y * h)
                x2, y2 = int(index.x * w), int(index.y * h)

                cv2.circle(frame, (x1, y1), 8, (0, 0, 225), -1)  # Blue filled circle
                cv2.circle(frame, (x2, y2), 8, (0, 0, 255), -1)

                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                distance = math.hypot(x2 - x1, y2 - y1)

                if distance < 30:
                    gesture_name = "Pinch / Click"
                elif distance < 80:
                    gesture_name = "Near / Hold"
                else:
                    gesture_name = "Open / Spread"

                break

        cv2.putText(frame, f"Distance: {int(distance)} px", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(frame, f"Gesture: {gesture_name}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(frame, "Press 'S' = Screenshot | Press 'Q' = Quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2)

        cv2.imshow("Milestone 2 ", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            ts = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(OUTPUT_DIR, f"gesture_{ts}.png")
            cv2.imwrite(filename, frame)
            print("Screenshot saved:", filename)

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
