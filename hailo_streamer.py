# region imports
import os
os.environ["GST_PLUGIN_FEATURE_RANK"] = "vaapidecodebin:NONE"

import time
import threading
from pathlib import Path
from datetime import datetime

import cv2
import gi

gi.require_version("Gst", "1.0")

import hailo
from flask import Flask, Response

from hailo_apps.python.pipeline_apps.detection_simple.detection_simple_pipeline import (
    GStreamerDetectionSimpleApp,
)
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)

hailo_logger = get_logger(__name__)
# endregion imports

app = Flask(__name__)

latest_jpeg = None
jpeg_lock = threading.Lock()

# Stream tuning
STREAM_WIDTH = 1080
STREAM_HEIGHT = 720
JPEG_QUALITY = 85
STREAM_EVERY_N_FRAMES = 1

# Recording settings
SNAPSHOT_DIR = Path("snapshots")
SNAPSHOT_DIR.mkdir(exist_ok=True)

CAT_CONFIDENCE_THRESHOLD = 0.3

# Start after cat is present for this long
CAT_START_DETECTION_SECONDS = 1.0

# Stop after cat has been gone this long
CAT_LOST_TIMEOUT = 5.0

RECORD_FPS = 10

recording = False
video_writer = None
current_recording_path = None

last_cat_seen = 0.0
cat_detection_start_time = None
last_debug_print = 0.0


def get_day_snapshot_dir():
    day_str = datetime.now().strftime("%Y-%m-%d")
    day_dir = SNAPSHOT_DIR / day_str
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir


def start_recording(width, height):
    global recording, video_writer, current_recording_path

    if recording:
        return

    day_dir = get_day_snapshot_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_recording_path = day_dir / f"cat_visit_{timestamp}.mp4"

    print(f"[DEBUG] Attempting to start recording: {current_recording_path}")
    print(f"[DEBUG] Writer size: {width}x{height}, fps={RECORD_FPS}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        str(current_recording_path),
        fourcc,
        RECORD_FPS,
        (width, height)
    )

    if not video_writer.isOpened():
        print(f"[ERROR] Failed to start recording: {current_recording_path}")
        video_writer = None
        current_recording_path = None
        recording = False
        return

    recording = True
    print(f"[INFO] Recording started: {current_recording_path}")


def stop_recording():
    global recording, video_writer, current_recording_path

    if video_writer is not None:
        video_writer.release()
        video_writer = None

    if recording and current_recording_path is not None:
        print(f"[INFO] Recording saved: {current_recording_path}")

    current_recording_path = None
    recording = False


class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.use_frame = True


def app_callback(element, buffer, user_data):
    global latest_jpeg
    global last_cat_seen, recording, video_writer, cat_detection_start_time, last_debug_print

    frame_idx = user_data.get_count()

    if buffer is None:
        return

    pad = element.get_static_pad("src")
    if pad is None:
        return

    fmt, width, height = get_caps_from_pad(pad)
    if fmt is None or width is None or height is None:
        return

    now = time.time()

    roi = hailo.get_roi_from_buffer(buffer)
    cat_detected_this_frame = False
    cat_boxes = []
    max_cat_conf = 0.0

    for detection in roi.get_objects_typed(hailo.HAILO_DETECTION):
        label = detection.get_label()
        conf = detection.get_confidence()

        if label != "cat" or conf < CAT_CONFIDENCE_THRESHOLD:
            continue

        cat_detected_this_frame = True
        max_cat_conf = max(max_cat_conf, conf)

        bbox = detection.get_bbox()
        x1 = int(bbox.xmin() * width)
        y1 = int(bbox.ymin() * height)
        x2 = int(bbox.xmax() * width)
        y2 = int(bbox.ymax() * height)

        cat_boxes.append((x1, y1, x2, y2, conf))

    if cat_detected_this_frame:
        last_cat_seen = now

        if cat_detection_start_time is None:
            cat_detection_start_time = now
            print(f"[DEBUG] Cat detection streak started, conf={max_cat_conf:.2f}")

        detection_duration = now - cat_detection_start_time

        if now - last_debug_print > 0.5:
            print(
                f"[DEBUG] cat detected, conf={max_cat_conf:.2f}, "
                f"streak={detection_duration:.2f}s, recording={recording}"
            )
            last_debug_print = now

        if (not recording) and (detection_duration >= CAT_START_DETECTION_SECONDS):
            start_recording(STREAM_WIDTH, STREAM_HEIGHT)
    else:
        if not recording:
            if cat_detection_start_time is not None:
                print("[DEBUG] Cat streak reset before recording started")
            cat_detection_start_time = None

    if recording and (now - last_cat_seen > CAT_LOST_TIMEOUT):
        print("[DEBUG] Cat lost timeout reached, stopping recording")
        stop_recording()
        cat_detection_start_time = None

    need_stream_frame = (frame_idx % STREAM_EVERY_N_FRAMES == 0)
    need_record_frame = recording

    if not need_stream_frame and not need_record_frame:
        return

    frame = get_numpy_from_buffer(buffer, fmt, width, height)
    if frame is None:
        return

    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        frame_bgr = frame.copy()

    for x1, y1, x2, y2, conf in cat_boxes:
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame_bgr,
            f"cat {conf:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        frame_bgr,
        timestamp,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    status_text = "RECORDING" if recording else "IDLE"
    cv2.putText(
        frame_bgr,
        status_text,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255) if recording else (200, 200, 200),
        2,
    )

    frame_bgr = cv2.resize(frame_bgr, (STREAM_WIDTH, STREAM_HEIGHT))

    if recording and video_writer is not None:
        video_writer.write(frame_bgr)

    if need_stream_frame:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        ok, buffer_jpg = cv2.imencode(".jpg", frame_bgr, encode_params)
        if ok:
            with jpeg_lock:
                latest_jpeg = buffer_jpg.tobytes()


def mjpeg_generator():
    global latest_jpeg

    while True:
        with jpeg_lock:
            frame = latest_jpeg

        if frame is None:
            time.sleep(0.01)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )

        time.sleep(0.03)


@app.route("/")
def index():
    return f"""
    <html>
      <head>
        <title>Hailo Detection Stream</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body style="background:#111; color:#eee; font-family:Arial; text-align:center; margin:0; padding:20px;">
        <h2>Hailo Detection Stream</h2>
        <img src="/video_feed" width="{STREAM_WIDTH}" style="max-width:95vw; height:auto; border:1px solid #444;" />
      </body>
    </html>
    """


@app.route("/video_feed")
def video_feed():
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def run_flask():
    app.run(host="0.0.0.0", port=5000, threaded=True)


def main():
    hailo_logger.info("Starting Detection Simple App with Flask stream.")
    user_data = user_app_callback_class()

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    hailo_logger.info("Open browser at http://<pi-ip>:5000")
    gst_app = GStreamerDetectionSimpleApp(app_callback, user_data)

    try:
        gst_app.run()
    finally:
        stop_recording()


if __name__ == "__main__":
    main()