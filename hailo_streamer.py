# region imports
import os
os.environ["GST_PLUGIN_FEATURE_RANK"] = "vaapidecodebin:NONE"

import time
import threading
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
STREAM_WIDTH = 640
STREAM_HEIGHT = 480
JPEG_QUALITY = 60
STREAM_EVERY_N_FRAMES = 3  # larger = less lag, lower FPS in browser


class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.use_frame = True


def app_callback(element, buffer, user_data):
    global latest_jpeg

    frame_idx = user_data.get_count()

    if buffer is None:
        return

    # Skip frames for the web stream to reduce lag
    if frame_idx % STREAM_EVERY_N_FRAMES != 0:
        return

    pad = element.get_static_pad("src")
    if pad is None:
        return

    fmt, width, height = get_caps_from_pad(pad)
    if fmt is None or width is None or height is None:
        return

    frame = None
    if user_data.use_frame:
        frame = get_numpy_from_buffer(buffer, fmt, width, height)

    if frame is None:
        return

    # Convert to BGR for OpenCV drawing/encoding
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        frame_bgr = frame.copy()

    # Draw detections
    # Draw detections
    roi = hailo.get_roi_from_buffer(buffer)
    for detection in roi.get_objects_typed(hailo.HAILO_DETECTION):
        label = detection.get_label()
        conf = detection.get_confidence()

        # Only keep cats above threshold
        if label != "cat" or conf < 0.4:
            continue

        bbox = detection.get_bbox()

        x1 = int(bbox.xmin() * width)
        y1 = int(bbox.ymin() * height)
        x2 = int(bbox.xmax() * width)
        y2 = int(bbox.ymax() * height)

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame_bgr,
            f"{label} {conf:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # Timestamp
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

    # Resize smaller for less delay
    frame_bgr = cv2.resize(frame_bgr, (STREAM_WIDTH, STREAM_HEIGHT))

    # Lower JPEG quality for lower latency
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    ok, buffer_jpg = cv2.imencode(".jpg", frame_bgr, encode_params)
    if not ok:
        return

    with jpeg_lock:
        latest_jpeg = buffer_jpg.tobytes()


def mjpeg_generator():
    global latest_jpeg
    last_sent = None

    while True:
        with jpeg_lock:
            frame = latest_jpeg

        if frame is None:
            time.sleep(0.01)
            continue

        # Do not resend identical frame over and over as fast as possible
        if frame == last_sent:
            time.sleep(0.01)
            continue

        last_sent = frame

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


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
    gst_app.run()


if __name__ == "__main__":
    main()