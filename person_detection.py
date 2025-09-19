import cv2
from ultralytics import YOLO
import numpy as np
import simpleaudio as sa
import time
import csv
import os

# --- Configuration ---
CAMERA_INDEX = 0
MODEL_PATH = "yolov8n.pt"
MIN_AREA_RATIO = 0.03
GLOBAL_BEEP_COOLDOWN = 10  # seconds
CSV_FILE = "visitor_log.csv"

# --- Beep Sound ---
def play_beep(frequency=440, duration=0.3, volume=0.3):
    fs = 44100
    t = np.linspace(0, duration, int(fs * duration), False)
    wave = np.sin(frequency * 2 * np.pi * t)
    audio = (wave * volume * 32767).astype(np.int16)
    sa.play_buffer(audio, 1, 2, fs)  # Non-blocking

# --- Initialize CSV ---
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Person_ID", "First_Seen", "Last_Seen", "Duration_sec"])

# --- Load model and camera ---
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

cv2.namedWindow("Person Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Person Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# --- Tracking variables ---
current_ids = set()
last_beep_time = 0  # global cooldown
person_times = {}   # track_id -> {"first_seen": timestamp, "last_seen": timestamp}
completed_durations = []  # list of durations for finished visits

# --- Main loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_area = frame.shape[0] * frame.shape[1]

    # Run YOLO tracking
    results = model.track(frame, classes=[0], persist=True, verbose=False)
    new_ids = set()
    boxes = results[0].boxes

    if boxes.id is not None:
        for idx, track_id in enumerate(boxes.id.cpu().numpy().astype(int)):
            x1, y1, x2, y2 = boxes.xyxy[idx].cpu().numpy()
            box_area = (x2 - x1) * (y2 - y1)
            if box_area / frame_area >= MIN_AREA_RATIO:
                new_ids.add(track_id)
                if track_id not in person_times:
                    person_times[track_id] = {"first_seen": time.time(), "last_seen": time.time()}
                else:
                    person_times[track_id]["last_seen"] = time.time()

    # --- Global cooldown beep ---
    newcomers = new_ids - current_ids
    current_time = time.time()
    if newcomers and (current_time - last_beep_time >= GLOBAL_BEEP_COOLDOWN):
        print(f"[INFO] New person(s) detected! IDs: {newcomers}")
        play_beep()
        last_beep_time = current_time

    # --- Check for people who left frame ---
    left_ids = current_ids - new_ids
    for track_id in left_ids:
        first = person_times[track_id]["first_seen"]
        last = person_times[track_id]["last_seen"]
        duration = round(last - first, 2)
        completed_durations.append(duration)
        with open(CSV_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([track_id,
                             time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(first)),
                             time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last)),
                             duration])
        person_times.pop(track_id)

    current_ids = new_ids
    person_count = len(current_ids)
    total_unique = len(completed_durations) + len(current_ids)
    avg_duration = round(np.mean(completed_durations), 2) if completed_durations else 0

    # Annotate frame
    annotated_frame = results[0].plot()
    cv2.putText(annotated_frame, f"Current in Frame: {person_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Total Unique Visitors: {total_unique}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"Avg Duration: {avg_duration} sec", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display frame
    cv2.imshow("Person Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
# Log remaining people still in frame
for track_id in list(person_times.keys()):
    first = person_times[track_id]["first_seen"]
    last = person_times[track_id]["last_seen"]
    duration = round(last - first, 2)
    completed_durations.append(duration)
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([track_id,
                         time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(first)),
                         time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last)),
                         duration])

cap.release()
cv2.destroyAllWindows()
