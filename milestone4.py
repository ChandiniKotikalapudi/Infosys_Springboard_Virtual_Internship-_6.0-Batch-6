import cv2
import mediapipe as mp
import math
import numpy as np
import time
from collections import deque

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

BASE_W, BASE_H = 1280, 720

# Initialize MediaPipe & Camera
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, BASE_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, BASE_H)

# OpenCV Window
WINDOW_NAME = "Milestone 4 - Gesture Volume Control Dashboard"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, BASE_W, BASE_H)

# Initialize PyCAW
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_min, vol_max = volume.GetVolumeRange()[:2]

# Smoothing & Graph History
volume_history = deque(maxlen=7)
graph_history = deque(maxlen=150)

last_vol_percent = 0
prev_time = 0

def S(x, y):
    return int(x * scale_x), int(y * scale_y)


def SS(v):
    return int(v * min(scale_x, scale_y))

# Main Loop
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    scale_x = w / BASE_W
    scale_y = h / BASE_H

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    status_text = "Show your hand"
    # We don't reset vol_percent to 0 here; we use the last known value
    current_vol_display = last_vol_percent

    # Hand Tracking & Volume Control
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),  # landmarks (green)
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)  # connections (green)
            )

            x1, y1 = lm_list[4][1], lm_list[4][2]  # Thumb tip
            x2, y2 = lm_list[8][1], lm_list[8][2]  # Index tip

            cv2.circle(img, (x1, y1), SS(10), (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), SS(10), (0, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), SS(3))

            distance = math.hypot(x2 - x1, y2 - y1)

            raw_volume = np.interp(distance, [20, 250], [vol_min, vol_max])
            volume_history.append(raw_volume)
            smooth_volume = sum(volume_history) / len(volume_history)
            volume.SetMasterVolumeLevel(smooth_volume, None)

            last_vol_percent = np.interp(smooth_volume, [vol_min, vol_max], [0, 100])
            current_vol_display = last_vol_percent

            if current_vol_display < 10:
                status_text = "Muted"
            elif current_vol_display < 40:
                status_text = "Low Volume"
            elif current_vol_display < 70:
                status_text = "Medium Volume"
            else:
                status_text = "High Volume"
            break

    graph_history.append(current_vol_display)

    # Dashboard UI Panels
    # -------------------------------
    # Side Panel
    cv2.rectangle(img, S(0, 0), S(300, 720), (40, 40, 40), cv2.FILLED)

    # Title
    cv2.putText(img, "Gesture Volume", S(30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1 * min(scale_x, scale_y),
                (0, 255, 255), SS(2))
    cv2.putText(img, "Control Dashboard", S(30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8 * min(scale_x, scale_y),
                (255, 255, 255), SS(2))

    # Volume Bar
    bar_top, bar_bottom = 180, 600
    bar_x1, _ = S(80, bar_top)
    bar_x2, _ = S(120, bar_bottom)
    y_top, y_bottom = int(bar_top * scale_y), int(bar_bottom * scale_y)

    cv2.rectangle(img, (bar_x1, y_top), (bar_x2, y_bottom), (0, 255, 255), SS(3))
    vol_bar = np.interp(current_vol_display, [0, 100], [bar_bottom, bar_top])
    y_bar = int(vol_bar * scale_y)
    cv2.rectangle(img, (bar_x1, y_bar), (bar_x2, y_bottom), (0, 255, 0), cv2.FILLED)

    cv2.putText(img, f"{int(current_vol_display)} %", S(60, 650),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9 * min(scale_x, scale_y),
                (255, 255, 255), SS(2))

    # LIVE GRAPH PANEL
    graph_x_origin = 350
    graph_y_origin = 650
    graph_height = 150
    graph_width = 400

    cv2.rectangle(img, S(graph_x_origin, graph_y_origin - graph_height - 20),
                  S(graph_x_origin + graph_width + 10, graph_y_origin + 20), (30, 30, 30), cv2.FILLED)
    cv2.putText(img, "Live Volume Level (History)", S(graph_x_origin + 10, graph_y_origin - graph_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * min(scale_x, scale_y), (0, 255, 255), SS(1))

    for i in range(1, len(graph_history)):
        pt1 = S(graph_x_origin + (i - 1) * 2.5, graph_y_origin - (graph_history[i - 1] * 1.5))
        pt2 = S(graph_x_origin + i * 2.5, graph_y_origin - (graph_history[i] * 1.5))
        cv2.line(img, pt1, pt2, (0, 255, 0), SS(2))


    # Status & FPS
    cv2.putText(img, f"Status: {status_text}", S(40, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7 * min(scale_x, scale_y),
                (0, 255, 0), SS(2))

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
    prev_time = curr_time
    cv2.putText(img, f"FPS: {int(fps)}", S(40, 700),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7 * min(scale_x, scale_y),
                (0, 255, 255), SS(2))

    cv2.putText(img, "Pinch thumb & index finger to control volume | Press Q to Quit",
                S(350, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * min(scale_x, scale_y),
                (255, 255, 255), SS(2))


    cv2.imshow(WINDOW_NAME, img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()