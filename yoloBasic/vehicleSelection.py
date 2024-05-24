from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLOv8 model with tracking capability
model = YOLO('../yolov8m.pt', task='track')  # Ensure the model is set for tracking

# Global variable to store the selected tracking ID
selected_track_id = None
objects = []

# Mouse callback function
def select_object(event, x, y, flags, param):
    global selected_track_id, objects
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click is inside any object bounding box
        for obj in objects:
            track_id, bbox, label = obj
            if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                selected_track_id = track_id
                print(f"Selected {label} ID: {selected_track_id}")
                break

# Initialize video capture
cap = cv2.VideoCapture('videos/los_angeles.mp4')  # Replace with the path to your video file

# Create a named window and set a mouse callback
cv2.namedWindow('Tracking')
cv2.setMouseCallback('Tracking', select_object)

# Track objects
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform tracking
    results = model.track(source=frame, persist=True)  # persist=True to maintain tracking across frames

    # Clear the objects list for the current frame
    objects = []

    # Iterate through the results to extract bounding boxes and IDs
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i])
                track_id = int(boxes.id[i]) if boxes.id is not None else None
                bbox = boxes.xyxy[i].cpu().numpy()  # Bounding box in xyxy format

                # Get the label for the class ID
                label = model.names[class_id]

                # Check if the detected object belongs to any of the desired classes
                if label in ['person', 'animal', 'car', 'truck', 'bus']:  # Add more classes as needed
                    objects.append((track_id, bbox, label))

    # Draw bounding box only for the selected object
    if selected_track_id is not None:
        for track_id, bbox, label in objects:
            if track_id == selected_track_id:
                color = (0, 0, 255) if label == 'person' else (0, 255, 0)  # Red for person, green for others
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.putText(frame, f'ID: {selected_track_id} ({label})', (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                break
    
    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
