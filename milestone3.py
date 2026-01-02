import cv2
import mediapipe as mp
import math
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume_interface = cast(interface, POINTER(IAudioEndpointVolume))
vol_min, vol_max = volume_interface.GetVolumeRange()[:2]


plt.ioff()
fig, ax = plt.subplots(figsize=(6, 2.5), dpi=100)
canvas = FigureCanvas(fig)

volume_history = []
line, = ax.plot([], [], lw=2)
ax.set_ylim(0, 100)
ax.set_title("Volume Level (%)")
ax.set_xlabel("Frames")
ax.set_ylabel("Volume")

def get_graph_image():
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba())
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return img

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

OUTPUT_DIR = "screenshots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Check permissions.")


with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
) as hands:

    prev_volume = 0
    smoothed_volume = 0

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
        volume_percent = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

                thumb = hand_landmarks.landmark[4]
                index = hand_landmarks.landmark[8]

                x1, y1 = int(thumb.x * w), int(thumb.y * h)
                x2, y2 = int(index.x * w), int(index.y * h)

                cv2.circle(frame, (x1, y1), 8, (0, 0, 255), -1)
                cv2.circle(frame, (x2, y2), 8, (0, 0, 255), -1)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                distance = math.hypot(x2 - x1, y2 - y1)

                if distance < 30:
                    gesture_name = "Pinch / Click"
                elif distance < 80:
                    gesture_name = "Near / Hold"
                else:
                    gesture_name = "Open / Spread"

                volume_percent = np.interp(distance, [20, 200], [0, 100])
                volume_percent = int(np.clip(volume_percent, 0, 100))

                target_volume = np.interp(volume_percent, [0, 100], [vol_min, vol_max])

                smoothed_volume = prev_volume + (target_volume - prev_volume) * 0.1
                prev_volume = smoothed_volume

                volume_interface.SetMasterVolumeLevel(smoothed_volume, None)

                volume_history.append(volume_percent)

                break


        cv2.putText(frame, f"Distance: {int(distance)} px", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(frame, f"Gesture: {gesture_name}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(frame, "Press 'S' = Screenshot | Press 'Q' = Quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2)


        bar_x1, bar_y1 = 560, 100
        bar_x2, bar_y2 = 580, 400

        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (0, 255, 255), 2)

        bar_height = int(np.interp(volume_percent, [0, 100], [bar_y2, bar_y1]))
        cv2.rectangle(frame, (bar_x1, bar_height), (bar_x2, bar_y2), (0, 255, 0), -1)

        cv2.putText(frame, f"{volume_percent}%", (bar_x1 - 15, bar_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


        line.set_xdata(np.arange(len(volume_history)))
        line.set_ydata(volume_history)
        ax.set_xlim(0, max(50, len(volume_history)))

        graph_img = get_graph_image()
        graph_img = cv2.resize(graph_img, (frame.shape[1], 250))

        combined = np.vstack((frame, graph_img))

        cv2.namedWindow("Milestone 3 - Gesture Volume Control + Graph", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Milestone 3 - Gesture Volume Control + Graph", 1300, 900)

        cv2.imshow("Milestone 3 - Gesture Volume Control + Graph", combined)


        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            ts = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(OUTPUT_DIR, f"gesture_{ts}.png")
            cv2.imwrite(filename, combined)
            print("Screenshot saved:", filename)

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
