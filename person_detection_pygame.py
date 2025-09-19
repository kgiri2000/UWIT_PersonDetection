import cv2
from ultralytics import YOLO
import numpy as np
import pygame  # Replaced winsound
import time
import csv
import os

# Configuration
CAMERA_INDEX = 0
MODEL_PATH = "yolov8n.pt"
MIN_AREA_RATIO = 0.03
#Global cooldown, later we can implement, per person cool down.
GLOBAL_BEEP_COOLDOWN = 10  # seconds
CSV_FILE = "visitor_log.csv"
SOUND_FILE = "notification.wav"  # use your WAV file
active_message = []
MESSAGE_DURATION = 5

# Initialize Pygame mixer for sound
pygame.mixer.init()
beep_sound = pygame.mixer.Sound(SOUND_FILE)

# Play sound
def play_beep():
    """
    Play the WAV file once (non-blocking).
    """
    beep_sound.play()

# Initialize CSV
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Person_ID", "First_Seen", "Last_Seen", "Duration_sec"])

# Load model and camera
# Current YOLOv8 model
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

# Create full screen
cv2.namedWindow("Person Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Person Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Tracking variables
current_ids = set()
 # global cooldown
last_beep_time = 0 
 # track_id -> {"first_seen": timestamp, "last_seen": timestamp}
beeped_ids = set()
BEEP_DELAY = 5
person_times = {}   
 # list of durations for finished visits
completed_durations = [] 

# Main loop
while True:
    #ret is boolean( True if frame is successfully read)
    #frame = current image of the camera( 2D array of pixels ( 460,640,3))
    ret, frame = cap.read()
    #if not captured, skip the iteration
    if not ret:
        continue
    
    frame_area = frame.shape[0] * frame.shape[1]

    # Run YOLO tracking
    #Track a person, class 0
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

    # Global cooldown beep
    newcomers = new_ids - current_ids
    current_time = time.time()
    # Don't beep right away, wait for 5 sec for each id
    # if newcomers and (current_time - last_beep_time >= GLOBAL_BEEP_COOLDOWN):
    #     print(f"[INFO] New person(s) detected! IDs: {newcomers}")
    #     play_beep()  # play WAV file
    #     last_beep_time = current_time
    '''
    If person stays with same id for 5 sec, it will notify
    Previous version would notify for each new ids
    Since, this is deep model, same person was getting different ids in differnt frames
    rendering a new id.

    Later work: Stronger re-identification tracker like: DeepSort or ByteTrack
    '''
    for track_id in new_ids:
        if track_id in person_times:
            first_seen = person_times[track_id]["first_seen"]
            duration = current_time - first_seen
             #Check if the person is new enough, 
            if(duration >= BEEP_DELAY and
                track_id not in beeped_ids and
                (current_time - last_beep_time >= GLOBAL_BEEP_COOLDOWN)):
                print(f"[INFO] Person {track_id} confirmed after {BEEP_DELAY} sec")
                msg = f"Person {track_id} confirmed after {BEEP_DELAY} sec."
                play_beep()
                last_beep_time = current_time
                '''
                Adding the person to the beeped id, later this can cause stack overflow.
                Need to find the better solution
                '''
                beeped_ids.add(track_id)
                active_message.append((msg, time.time()))


    # Check for people who left frame
    left_ids = current_ids - new_ids
    #Only record if they stay for more than 10 sec
    for track_id in left_ids:
        first = person_times[track_id]["first_seen"]
        last = person_times[track_id]["last_seen"]
        duration = round(last - first, 2)
        if duration >= 5:
            completed_durations.append(duration)
            with open(CSV_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([track_id,
                                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(first)),
                                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last)),
                                duration])

        person_times.pop(track_id)
        beeped_ids.discard(track_id)

    current_ids = new_ids
    person_count = len(current_ids)
    total_unique = len(completed_durations) + len(current_ids)
    avg_duration = round(np.mean(completed_durations), 2) if completed_durations else 0

    # Annotate frame
    annotated_frame = results[0].plot()
    cv2.putText(annotated_frame, f"Current in Frame: {person_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(annotated_frame, f"Total Unique Visitors: {total_unique}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(annotated_frame, f"Avg Duration: {avg_duration} sec", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    #Display the message
    current_time = time.time()
    y_offset = annotated_frame.shape[0]-20
    for msg, ts in list(active_message):
        if current_time - ts < MESSAGE_DURATION:
            cv2.putText(annotated_frame, msg, (20, y_offset),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, (0,0,0), 1)
            #stack message vertically
            y_offset -=20 
        else:
            active_message.remove((msg, ts))

    # Display frame
    cv2.imshow("Person Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
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
pygame.mixer.quit()  #clean up sound
