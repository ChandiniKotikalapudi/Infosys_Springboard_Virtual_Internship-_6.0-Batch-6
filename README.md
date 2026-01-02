# 🎓 Infosys Springboard Virtual Internship

## Gesture-Based Volume Control Using Hand Gestures

---

## 📌 Project Overview

This project implements a **gesture-based system volume control application** using **computer vision**. By using a webcam and hand gestures, users can control the system’s audio volume without any physical contact.

The application detects hand landmarks in real time using **MediaPipe**, calculates the distance between the thumb and index finger, maps this distance to system volume levels using **PyCAW**, and displays the results on an interactive dashboard built with **OpenCV**.

---

## 🛠️ Technologies Used

* **Python**
* **OpenCV** – Video capture and UI rendering
* **MediaPipe Hands** – Hand landmark detection
* **PyCAW** – System volume control (Windows)
* **NumPy** – Numerical computations
* **Collections (Deque)** – Smoothing and graph history

---

## 🧩 Milestone-wise Description

### 🟢 Milestone 1: Hand Detection

* Captured live video using webcam
* Detected hands using MediaPipe
* Identified 21 hand landmarks per hand
* Displayed landmarks and connections in real time

---

### 🟡 Milestone 2: Gesture Recognition

* Tracked thumb tip and index finger tip
* Calculated distance between fingers
* Used gesture distance as input control
* Visualized gesture on the screen

---

### 🔵 Milestone 3: Volume Control

* Mapped gesture distance to system volume range
* Controlled system volume using PyCAW
* Added smoothing to avoid abrupt volume changes
* Displayed real-time volume percentage

---

### 🔴 Milestone 4: UI Dashboard & Feedback

* Designed a dark-themed dashboard UI
* Added volume bar and live volume graph
* Displayed system status and FPS
* Implemented responsive UI scaling



## 🎮 How It Works

1. The webcam captures live video
2. MediaPipe detects hand landmarks
3. Distance between thumb and index finger is calculated
4. Distance is mapped to system volume
5. Volume is updated smoothly using PyCAW
6. UI dashboard displays volume, status, and graph

---

## ✋ User Instructions

* Show one hand in front of the webcam
* Pinch thumb and index finger to control volume
* Increase distance → Increase volume
* Decrease distance → Decrease volume
* Press **Q** to quit the application

---

## 🏁 Conclusion

This project successfully demonstrates a **touchless volume control system** using hand gestures. The milestone-based development approach helped build the application step by step, resulting in a complete and user-friendly solution aligned with the **Infosys Springboard Virtual Internship** objectives.
