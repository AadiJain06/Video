import cv2
import numpy as np
from ultralytics import YOLO
import time
import math
import os
import datetime
from collections import deque

# Create output directories
os.makedirs("output", exist_ok=True)
os.makedirs("output/recordings", exist_ok=True)
os.makedirs("output/screenshots", exist_ok=True)
os.makedirs("output/summaries", exist_ok=True)

# ------------------- INITIALIZATION -------------------

model = YOLO("models/yolov8m.pt")  # Consider yolov8n.pt for speed or yolov8l.pt for accuracy
PERSON_LABEL = "person"
DISTRACTING_OBJECTS = ["cell phone", "laptop", "tv", "remote", "bottle", "wine glass", "cup"]
PROFESSIONAL_BACKGROUND = ["chair", "potted plant", "book", "clock"]
INTERVIEW_OBJECTS = ["book", "laptop", "keyboard", "mouse", "cup", "bottle", "cell phone", "document"]
ELECTRONIC_DEVICES = ["cell phone", "laptop", "tablet", "smart watch", "earbuds"]
PROHIBITED_ITEMS = ["cell phone", "wine glass", "beer", "cigarette", "knife", "scissors", "tablet", "smart watch", "earbuds"]
CONFIDENCE_THRESHOLD = 0.5  # Increased for higher precision
FRAME_SKIP = 2  # Process every 2nd frame for efficiency

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
output_path = "output/output.mp4"
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

last_classification = "Unknown"
classification_history = deque(maxlen=5)  # For temporal consistency
frame_brightness_history = deque(maxlen=30)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
posture_history = deque(maxlen=30)
attention_score = 100.0
last_face_position = None
recommendations = []
start_time = time.time()
frames_without_face = 0
background_consistency = deque(maxlen=30)
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)

interview_segments = []
current_segment_start = time.time()
current_segment_frames = []
is_recording_segment = True
segment_counter = 0
segment_event_log = []

identity_verified = False
prohibited_items_detected = []
distractions_log = []

metrics_history = {
    "attention": deque(maxlen=100),
    "posture": deque(maxlen=100),
    "professionalism": deque(maxlen=100)
}

# ------------------- HELPER FUNCTIONS -------------------

def create_dashboard(frame, metrics):
    dashboard_width = frame_width // 4
    dashboard = np.ones((frame_height, dashboard_width, 3), dtype=np.uint8) * 240
    
    cv2.putText(dashboard, "INTERVIEW METRICS", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.line(dashboard, (10, 40), (dashboard_width-10, 40), (180, 180, 180), 2)
    
    y_offset = 70
    cv2.putText(dashboard, f"Professionalism: {metrics['professionalism']:.1f}%", (15, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    bar_length = dashboard_width - 30
    filled_length = int(bar_length * metrics['professionalism'] / 100)
    cv2.rectangle(dashboard, (15, y_offset+10), (15+bar_length, y_offset+20), (200, 200, 200), -1)
    cv2.rectangle(dashboard, (15, y_offset+10), (15+filled_length, y_offset+20), 
                  (0, 128, 0) if metrics['professionalism'] > 70 else (0, 0, 255), -1)
    
    y_offset += 40
    cv2.putText(dashboard, f"Attention: {metrics['attention']:.1f}%", (15, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    filled_length = int(bar_length * metrics['attention'] / 100)
    cv2.rectangle(dashboard, (15, y_offset+10), (15+bar_length, y_offset+20), (200, 200, 200), -1)
    cv2.rectangle(dashboard, (15, y_offset+10), (15+filled_length, y_offset+20), 
                  (0, 128, 0) if metrics['attention'] > 70 else (0, 0, 255), -1)
    
    y_offset += 40
    cv2.putText(dashboard, f"Posture: {metrics['posture']:.1f}%", (15, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    filled_length = int(bar_length * metrics['posture'] / 100)
    cv2.rectangle(dashboard, (15, y_offset+10), (15+bar_length, y_offset+20), (200, 200, 200), -1)
    cv2.rectangle(dashboard, (15, y_offset+10), (15+filled_length, y_offset+20), 
                  (0, 128, 0) if metrics['posture'] > 70 else (0, 0, 255), -1)
    
    y_offset += 50
    session_duration = time.time() - start_time
    minutes, seconds = divmod(int(session_duration), 60)
    cv2.putText(dashboard, f"Duration: {minutes:02d}:{seconds:02d}", (15, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    y_offset += 25
    cv2.putText(dashboard, f"Segments: {len(interview_segments)}", (15, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    y_offset += 25
    cv2.putText(dashboard, f"People: {metrics['person_count']}", (15, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return dashboard

def start_new_segment():
    global current_segment_start, current_segment_frames, segment_counter, is_recording_segment
    if current_segment_frames:
        segment_path = f"output/recordings/segment_{segment_counter:03d}.mp4"
        segment_writer = cv2.VideoWriter(segment_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
        for frame in current_segment_frames:
            segment_writer.write(frame)
        segment_writer.release()
        segment_duration = time.time() - current_segment_start
        interview_segments.append({
            "id": segment_counter,
            "path": segment_path,
            "start_time": current_segment_start,
            "duration": segment_duration,
            "events": segment_event_log.copy()
        })
        segment_counter += 1
    current_segment_start = time.time()
    current_segment_frames = []
    segment_event_log.clear()
    is_recording_segment = True

def add_segment_event(event_type, description):
    segment_event_log.append({
        "timestamp": time.time() - current_segment_start,
        "type": event_type,
        "description": description
    })

def save_interview_summary():
    summary_path = f"output/summaries/interview_summary_{int(start_time)}.txt"
    with open(summary_path, "w") as f:
        f.write("=== INTERVIEW SUMMARY ===\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        total_duration = time.time() - start_time
        minutes, seconds = divmod(int(total_duration), 60)
        f.write(f"Duration: {minutes:02d}:{seconds:02d}\n")
        f.write(f"Segments: {len(interview_segments)}\n\n")
        for idx, segment in enumerate(interview_segments):
            segment_time = datetime.datetime.fromtimestamp(segment["start_time"]).strftime('%H:%M:%S')
            segment_duration = segment["duration"]
            minutes, seconds = divmod(int(segment_duration), 60)
            f.write(f"Segment {idx+1} - {segment_time} (Duration: {minutes:02d}:{seconds:02d})\n")
            if segment["events"]:
                f.write("  Events:\n")
                for event in segment["events"]:
                    event_time = event["timestamp"]
                    minutes, seconds = divmod(int(event_time), 60)
                    f.write(f"    - [{minutes:02d}:{seconds:02d}] {event['type']}: {event['description']}\n")
            f.write("\n")
        if distractions_log:
            f.write("=== DISTRACTIONS ===\n")
            for distraction in distractions_log:
                distraction_time = datetime.datetime.fromtimestamp(distraction["timestamp"]).strftime('%H:%M:%S')
                f.write(f"[{distraction_time}] {distraction['type']}: {distraction['description']}\n")
    return summary_path

def generate_interview_report():
    report_path = f"output/summaries/interview_report_{int(start_time)}.html"
    avg_attention = np.mean(list(metrics_history["attention"])) if metrics_history["attention"] else 0
    avg_posture = np.mean(list(metrics_history["posture"])) if metrics_history["posture"] else 0
    avg_professionalism = np.mean(list(metrics_history["professionalism"])) if metrics_history["professionalism"] else 0
    
    with open(report_path, "w") as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Interview Session Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; }}
                h1, h2 {{ color: #2c3e50; }}
                .container {{ max-width: 800px; margin: 0 auto; padding: 20px; background-color: white; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                .metric {{ margin: 10px 0; }}
                .metric-label {{ font-weight: bold; color: #34495e; }}
                .metric-value {{ font-size: 20px; color: #2980b9; }}
                .segment {{ border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 5px; background-color: #fafafa; }}
                .event {{ margin: 5px 0; padding: 5px; background-color: #f0f0f0; border-left: 4px solid #e74c3c; }}
                .distraction {{ margin: 5px 0; padding: 5px; background-color: #ffe6e6; border-left: 4px solid #c0392b; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Interview Session Report</h1>
                <p><strong>Date:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Session Overview</h2>
                <div class="metric">
                    <span class="metric-label">Duration:</span> 
                    <span class="metric-value">{datetime.timedelta(seconds=int(time.time() - start_time))}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Average Professionalism:</span> 
                    <span class="metric-value">{avg_professionalism:.1f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Average Attention:</span> 
                    <span class="metric-value">{avg_attention:.1f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Average Posture:</span> 
                    <span class="metric-value">{avg_posture:.1f}%</span>
                </div>
                
                <h2>Segment Details</h2>
        """)
        
        for idx, segment in enumerate(interview_segments):
            segment_time = datetime.datetime.fromtimestamp(segment["start_time"]).strftime('%H:%M:%S')
            segment_duration = segment["duration"]
            f.write(f"""
                <div class="segment">
                    <h3>Segment {idx+1} - {segment_time}</h3>
                    <p>Duration: {datetime.timedelta(seconds=int(segment_duration))}</p>
                    <h4>Events:</h4>
            """)
            if segment["events"]:
                for event in segment["events"]:
                    event_time = datetime.timedelta(seconds=int(event["timestamp"]))
                    f.write(f"""
                        <div class="event">
                            <strong>[{event_time}]</strong> {event['type']}: {event['description']}
                        </div>
                    """)
            else:
                f.write("<p>No notable events</p>")
            f.write("</div>")
        
        if distractions_log:
            f.write("""
                <h2>Distractions and Violations</h2>
            """)
            for distraction in distractions_log:
                distraction_time = datetime.datetime.fromtimestamp(distraction["timestamp"]).strftime('%H:%M:%S')
                f.write(f"""
                    <div class="distraction">
                        <strong>[{distraction_time}]</strong> {distraction['type']}: {distraction['description']}
                    </div>
                """)
        
        f.write("""
            </div>
        </body>
        </html>
        """)
    return report_path

def check_prohibited_items(detected_objects):
    return [item for item in PROHIBITED_ITEMS if item in detected_objects]

def detect_phone_usage(detected_objects):
    return "cell phone" in detected_objects

def analyze_lighting(frame):
    brightness = cv2.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))[0]
    frame_brightness_history.append(brightness)
    if len(frame_brightness_history) > 5:
        variation = np.std(list(frame_brightness_history)[-5:])
        if brightness < 60:  # Adjusted for better sensitivity
            return "Low Light", brightness
        elif brightness > 180:  # Adjusted
            return "Too Bright", brightness
        elif variation > 15:  # Tighter threshold
            return "Inconsistent", brightness
        return "Good", brightness
    return "Good", brightness

def analyze_background(frame, detected_objects):
    fg_mask = background_subtractor.apply(frame)
    foreground_percentage = (np.count_nonzero(fg_mask) / (frame.shape[0] * frame.shape[1])) * 100
    background_consistency.append(foreground_percentage)
    avg_movement = np.mean(background_consistency) if background_consistency else 0
    distracting_items = [obj for obj in detected_objects if obj in DISTRACTING_OBJECTS]
    if len(distracting_items) > 0:
        return f"Distracting Items: {', '.join(distracting_items)}", foreground_percentage
    elif avg_movement > 10:  # Tighter threshold for stability
        return "Unstable Background", foreground_percentage
    else:
        professional_items = [obj for obj in detected_objects if obj in PROFESSIONAL_BACKGROUND]
        return "Professional Background" if professional_items else "Clean Background", foreground_percentage

def analyze_attention_posture(frame, face_coords=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))  # Adjusted for accuracy
    global last_face_position, attention_score, frames_without_face, posture_history

    if len(faces) == 0 or (face_coords and not any(x < fx < x+w and y < fy < y+h for (fx, fy, fw, fh) in faces for x, y, w, h in [face_coords])):
        frames_without_face += 1
        if frames_without_face > 5:  # Reduced threshold for quicker response
            attention_score = max(0.0, attention_score - 2.0)
            return "No Face Detected", 0.0, attention_score, None
        return "Face Lost", np.mean(posture_history) if posture_history else 0.0, attention_score, last_face_position
    
    frames_without_face = 0
    x, y, w, h = faces[0] if face_coords is None else face_coords
    roi_gray = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 4, minSize=(20, 20))  # Adjusted for precision
    eye_contact = len(eyes) >= 1

    # Posture: Refine with vertical position and size
    face_center_y = y + h / 2
    face_center_x = x + w / 2
    frame_center_x = frame_width / 2
    distance_from_center = math.sqrt((face_center_x - frame_center_x)**2) / frame_width
    posture_score = (1.0 - (face_center_y / frame_height)) * 100.0 * (h / frame_height)  # Factor in face size
    posture_score = min(100.0, max(0.0, posture_score * 1.5))  # Adjusted scaling
    posture_history.append(posture_score)
    avg_posture = np.mean(posture_history) if posture_history else posture_score

    # Attention: Smoother adjustment
    attention_delta = 0.5 if eye_contact else -0.5
    if distance_from_center > 0.25:  # Tighter centering requirement
        attention_delta -= 0.3
    attention_score = max(0.0, min(100.0, attention_score + attention_delta))

    posture_status = "Good" if avg_posture > 70 else "Poor"  # Higher threshold
    position_status = "Centered" if distance_from_center < 0.25 else "Off-Center"
    last_face_position = (x, y, w, h)
    return f"{position_status}, {posture_status}", avg_posture, attention_score, last_face_position

def generate_feedback(lighting_status, background_status, posture_attention_status, posture_score, attention_score, prohibited_items):
    recommendations = []
    if prohibited_items:
        recommendations.append(f"Remove: {', '.join(prohibited_items)}")
    if lighting_status != "Good":
        recommendations.append(f"Fix lighting: {lighting_status}")
    if "Distracting" in background_status:
        recommendations.append(background_status)
    elif "Unstable" in background_status:
        recommendations.append("Stabilize background")
    if "Poor" in posture_attention_status:
        recommendations.append("Sit upright")
    if "Off-Center" in posture_attention_status:
        recommendations.append("Center yourself")
    if "No Face" in posture_attention_status:
        recommendations.append("Stay in frame")
    if attention_score < 60:  # Higher threshold
        recommendations.append("Look at camera")
    return recommendations[:3]

# ------------------- MAIN PROCESSING LOOP -------------------

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        out.write(frame)
        continue

    results = model(frame)
    annotated_frame = frame.copy()

    detected_objects = set()
    person_count = 0
    person_boxes = []

    for box in results[0].boxes:
        if float(box.conf[0]) > CONFIDENCE_THRESHOLD:
            obj_name = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            detected_objects.add(obj_name)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (0, 255, 0) if obj_name in PROFESSIONAL_BACKGROUND else (0, 0, 255)
            if obj_name in PROHIBITED_ITEMS:
                color = (255, 0, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            label = f"{obj_name} {confidence:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if obj_name == PERSON_LABEL:
                person_count += 1
                coords = [x1, y1, x2 - x1, y2 - y1]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray[y1:y2, x1:x2], 1.3, 5, minSize=(50, 50))
                if len(faces) > 0:
                    person_boxes.append(coords)

    background_status, bg_change = analyze_background(frame, detected_objects)
    lighting_status, brightness = analyze_lighting(frame)
    face_coords = None if not person_boxes else person_boxes[0]
    posture_attention_status, posture_score, attention_score, face_coords = analyze_attention_posture(frame, face_coords)

    # Classification with temporal consistency
    electronic_devices_detected = [obj for obj in detected_objects if obj in ELECTRONIC_DEVICES]
    current_classification = "Unprofessional" if person_count > 1 or electronic_devices_detected else "Professional"
    classification_history.append(current_classification)
    last_classification = max(set(classification_history), key=classification_history.count)

    if person_count > 1:
        add_segment_event("multiple_people", f"Detected {person_count} people")
    if electronic_devices_detected:
        add_segment_event("electronic_device", f"Detected: {', '.join(electronic_devices_detected)}")
        distractions_log.append({"timestamp": time.time(), "type": "Electronic Device", "description": f"Detected: {', '.join(electronic_devices_detected)}"})

    prohibited_items = check_prohibited_items(detected_objects)
    if prohibited_items:
        add_segment_event("prohibited_items", f"Detected: {', '.join(prohibited_items)}")
        distractions_log.append({"timestamp": time.time(), "type": "Prohibited Item", "description": f"Detected: {', '.join(prohibited_items)}"})

    if detect_phone_usage(detected_objects):
        add_segment_event("distraction", "Phone usage detected")
        distractions_log.append({"timestamp": time.time(), "type": "Phone Usage", "description": "Phone usage detected"})

    if time.time() - current_segment_start > 60:
        start_new_segment()

    if is_recording_segment:
        current_segment_frames.append(annotated_frame.copy())

    professionalism_score = (
        (20 if person_count == 1 else 0) +
        (20 if lighting_status == "Good" else 0) +
        (20 if "Professional" in background_status or "Clean" in background_status else 0) +
        (posture_score / 5) +
        (attention_score / 5)
    ) - (10 * (len(prohibited_items) + (person_count - 1) if person_count > 1 else 0))
    professionalism_score = max(0, min(100, professionalism_score))

    metrics = {
        "professionalism": professionalism_score,
        "attention": attention_score,
        "posture": posture_score,
        "person_count": person_count,
        "objects": detected_objects
    }
    metrics_history["attention"].append(attention_score)
    metrics_history["posture"].append(posture_score)
    metrics_history["professionalism"].append(professionalism_score)

    current_recommendations = generate_feedback(lighting_status, background_status, posture_attention_status, 
                                               posture_score, attention_score, prohibited_items)

    dashboard = create_dashboard(frame, metrics)
    combined_frame = np.hstack((annotated_frame, dashboard))

    y_offset = 30
    for rec in current_recommendations:
        cv2.putText(combined_frame, rec, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_offset += 30

    status_text = f"Class: {last_classification} | Light: {lighting_status} | BG: {background_status.split(',')[0]}"
    cv2.putText(combined_frame, status_text, (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(combined_frame, timestamp, (frame_width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Interview Monitor', combined_frame)
    out.write(annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        screenshot_path = f"output/screenshots/screenshot_{int(time.time())}.jpg"
        cv2.imwrite(screenshot_path, combined_frame)
        print(f"Screenshot saved to {screenshot_path}")
    elif key == ord('n'):
        start_new_segment()
        print("Starting new segment")

# ------------------- CLEANUP -------------------

def cleanup():
    if is_recording_segment and current_segment_frames:
        start_new_segment()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    summary_path = save_interview_summary()
    print(f"Summary saved to {summary_path}")
    report_path = generate_interview_report()
    print(f"Report saved to {report_path}")

try:
    pass
except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    cleanup()