import cv2
import numpy as np
import onnxruntime as ort
import time
from picamera2 import Picamera2

# ==============================
# Load ONNX Model
# ==============================

session = ort.InferenceSession("model/best_int8.onnx")
input_name = session.get_inputs()[0].name

print("Model loaded successfully ?")

# ==============================
# Setup Pi Camera
# ==============================

picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(main={"size": (640, 480)})
)
picam2.start()

print("? Pothole Detection Started (Press Q to quit)")

# ==============================
# Logging
# ==============================

log_file = open("log.txt", "a")

while True:

    start_time = time.time()

    # Capture frame
    frame = picam2.capture_array()
    H, W, _ = frame.shape

    # ------------------------------
    # Fix RGBA ? RGB (important!)
    # ------------------------------
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # ------------------------------
    # Preprocessing (MATCH MODEL)
    # ------------------------------
    img = cv2.resize(frame, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    # ------------------------------
    # Run Inference
    # ------------------------------
    outputs = session.run(None, {input_name: img})
    detections = outputs[0][0]

    detected = False

    # ------------------------------
    # Draw Detections
    # ------------------------------
    for det in detections:

        conf = det[4]

        if conf > 0.5:

            detected = True

            x, y, w, h = det[:4]

            x1 = int((x - w/2) * W)
            y1 = int((y - h/2) * H)
            x2 = int((x + w/2) * W)
            y2 = int((y + h/2) * H)

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"Pothole {conf:.2f}"
            cv2.putText(frame, label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)

    # ------------------------------
    # FPS Counter
    # ------------------------------
    fps = 1 / (time.time() - start_time)

    cv2.putText(frame,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2)

    # ------------------------------
    # Save detection proof
    # ------------------------------
    if detected:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"detection_{timestamp}.jpg", frame)
        log_file.write(f"{timestamp} pothole detected\n")

    # ------------------------------
    # Show Window
    # ------------------------------
    cv2.imshow("Road Anomaly Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
log_file.close()
cv2.destroyAllWindows()
picam2.close()
