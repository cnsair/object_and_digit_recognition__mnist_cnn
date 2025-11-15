import threading
import time
import os
import cv2
import torch
from torchvision import models, transforms
import numpy as np

# ---- CONFIG ----
MODEL_NAME = "ssdlite320_mobilenet_v3_large"  # fast single-stage model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INFER_SIZE = 320          # ssdlite320 uses 320x320
SCORE_THRESHOLD = 0.5     # show detections above this score
FRAME_SKIP = 2            # process every Nth frame (1 = every frame; 2 = every other frame)
DRAW_BOX_THICKNESS = 2
SHOW_FPS = True
# ----------------

# COCO labels (common) - index matches torchvision detection labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'human', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Preprocess transform (only convert to tensor; models accept [0,1] floats)
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load model (SSDLite for speed). Falls back to Faster R-CNN only if unavailable.
def load_model():
    print("Loading model on device:", DEVICE)
    try:
        model = models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
    except Exception as e:
        print("ssdlite not available, falling back to fasterrcnn (slower). Error:", e)
        model = models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# Shared state between main thread (capture/display) and worker thread (inference)
state = {
    "frame": None,             # latest raw frame (BGR)
    "annotated": None,         # latest annotated frame to display (BGR)
    "running": True,
    "lock": threading.Lock(),
    "process_frame_id": 0
}

# Worker thread: performs inference on the latest frame when available
def inference_worker():
    last_processed_id = -1
    while state["running"]:
        # copy frame to local variable
        with state["lock"]:
            frame = state["frame"]
            frame_id = state["process_frame_id"]

        if frame is None or frame_id == last_processed_id:
            # nothing new to process
            time.sleep(0.005)
            continue

        # Optionally skip frames by checking frame_id % FRAME_SKIP
        if FRAME_SKIP > 1 and frame_id % FRAME_SKIP != 0:
            last_processed_id = frame_id
            continue

        # prepare image for model: resize to INFER_SIZE, convert to tensor, move to device
        h, w = frame.shape[:2]
        img_resized = cv2.resize(frame, (INFER_SIZE, INFER_SIZE))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            try:
                outputs = model(tensor)[0]
            except Exception as e:
                print("Model inference error:", e)
                last_processed_id = frame_id
                continue

        # parse outputs
        boxes = outputs.get("boxes").detach().cpu().numpy()
        scores = outputs.get("scores").detach().cpu().numpy()
        labels = outputs.get("labels").detach().cpu().numpy()

        # scale boxes back to original frame size
        scale_x = w / INFER_SIZE
        scale_y = h / INFER_SIZE

        annotated = frame.copy()
        for i, score in enumerate(scores):
            if score < SCORE_THRESHOLD:
                break
            box = boxes[i]
            x1, y1, x2, y2 = int(box[0] * scale_x), int(box[1] * scale_y), int(box[2] * scale_x), int(box[3] * scale_y)
            label_id = int(labels[i])
            label = COCO_INSTANCE_CATEGORY_NAMES[label_id] if label_id < len(COCO_INSTANCE_CATEGORY_NAMES) else str(label_id)
            text = f"{label}: {score:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), DRAW_BOX_THICKNESS)
            cv2.putText(annotated, text, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        # store annotated frame
        with state["lock"]:
            state["annotated"] = annotated
        last_processed_id = frame_id

# start worker thread
worker = threading.Thread(target=inference_worker, daemon=True)
worker.start()

# main loop: capture frames and display annotated frames (if available)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    state["running"] = False
    worker.join(timeout=1)
    raise SystemExit(1)

print("Press 'q' to quit, 'c' to capture a frame to ./captures/")

frame_counter = 0
fps_smooth = 0.0
t0 = time.time()
os.makedirs("captures", exist_ok=True)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam")
            break

        frame_counter += 1
        with state["lock"]:
            state["frame"] = frame.copy()
            state["process_frame_id"] += 1

        # display the annotated result if available, else raw frame
        with state["lock"]:
            disp = state["annotated"].copy() if state["annotated"] is not None else frame.copy()

        # show fps
        if SHOW_FPS:
            dt = time.time() - t0
            fps = 1.0 / max(dt, 1e-6)
            # smooth fps
            fps_smooth = 0.85 * fps_smooth + 0.15 * fps if fps_smooth else fps
            t0 = time.time()
            cv2.putText(disp, f"FPS: {fps_smooth:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow("Object Detection (fast)", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Quitting...")
            break
        if key == ord("c"):
            # save current displayed frame
            fname = os.path.join("captures", f"capture_{int(time.time())}.jpg")
            cv2.imwrite(fname, disp)
            print("Saved capture:", fname)

except KeyboardInterrupt:
    print("KeyboardInterrupt received — exiting")

finally:
    # stop worker and cleanup
    state["running"] = False
    worker.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()
    print("Exited cleanly.")
