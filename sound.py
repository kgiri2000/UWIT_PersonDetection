import winsound
import time

def play_notification_chime():
    """Plays a simple two-tone chime."""
    # First tone: Higher pitch, short duration
    winsound.Beep(800, 200)  # 800 Hz for 200 ms
    
    # A tiny pause is not strictly necessary but can make it sound more distinct
    # time.sleep(0.05) 
    
    # Second tone: Lower pitch, longer duration
    winsound.Beep(600, 300)  # 600 Hz for 300 ms

# --- Example Usage ---
print("Playing a notification chime...")
play_notification_chime()
print("Done.")