import cv2
import csv
import os
import time
from datetime import datetime

CAMERA_INDEX = 0
CHECK_INTERVAL = 1.0

SNAPSHOT_DIR = "snapshots"
CSV_PATH = "visits.csv"

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "snapshot_path"])

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def save_snapshot(frame, prefix):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SNAPSHOT_DIR, f"{prefix}_{ts}.jpg")
    ok = cv2.imwrite(path, frame)
    print(f"Saved: {path} | success={ok}")
    return path

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError("Could not open camera")

print("Camera opened successfully.")
print("Saving one snapshot every second. Press Ctrl+C to stop.")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame")
            time.sleep(CHECK_INTERVAL)
            continue

        print(f"[{now_str()}] Read frame: shape={frame.shape}")

        snapshot_path = save_snapshot(frame, "test")

        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([now_str(), snapshot_path])

        time.sleep(CHECK_INTERVAL)

finally:
    cap.release()
    print("Camera released.")