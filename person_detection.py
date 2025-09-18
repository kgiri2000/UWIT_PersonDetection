import cv2
from ultralytics import YOLO
import winsound  # Windows-safe beep

# Config
CAMERA_INDEX = 0
MODEL_PATH = "yolov8n.pt"
MIN_AREA_RATIO = 0.03  # minimum box area / frame area to count

# Load YOLO model
model = YOLO(MODEL_PATH)

# Open webcam
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

#Full-screen window
cv2.namedWindow("Person Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Person Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
print("[INFO] Webcam opened. Press 'q' to quit.")

#Tracking variables
current_ids = set()

def play_beep():
    """Play a short beep using Windows sound API."""
    winsound.Beep(800, 200)

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Failed to grab frame.")
        continue

    frame_area = frame.shape[0] * frame.shape[1]

    # Run YOLO tracking (class 0 = person)
    results = model.track(frame, classes=[0], persist=True)

    new_ids = set()
    boxes = results[0].boxes

    if boxes.id is not None:
        for idx, track_id in enumerate(boxes.id.cpu().numpy().astype(int)):
            # Check bounding box area
            x1, y1, x2, y2 = boxes.xyxy[idx].cpu().numpy()
            box_area = (x2 - x1) * (y2 - y1)
            if box_area / frame_area >= MIN_AREA_RATIO:
                new_ids.add(track_id)

    # Beep for new fully visible person
    added = new_ids - current_ids
    if len(added) > 0:
        play_beep()

    current_ids = new_ids
    person_count = len(current_ids)

    # Annotate frame
    annotated_frame = results[0].plot()
    cv2.putText(annotated_frame, f"Persons in Frame: {person_count}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show full-screen frame
    cv2.imshow("Person Detection", annotated_frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
