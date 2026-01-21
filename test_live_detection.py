from ultralytics import YOLO
import cv2

# Load the model
model = YOLO('/home/athul/wonderla/best (3).pt')

# Open the video
video_path = '/home/athul/wonderla/NVR_ch46_main_20251230150021_20251230153000.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {fps} FPS, {width}x{height}, Total frames: {total_frames}")
print("Processing... Press 'q' to quit")

frame_count = 0
skip_frames = 2  # Process every 3rd frame for speed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Skip frames for faster processing
    if frame_count % skip_frames != 0:
        continue
    
    # Run inference
    results = model(frame, conf=0.25, verbose=False)
    
    # Get annotated frame
    annotated_frame = results[0].plot()
    
    # Display detections info
    detections = len(results[0].boxes)
    cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames} | Detections: {detections}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show frame
    cv2.imshow('YOLO Detection', annotated_frame)
    
    # Print progress every 30 frames
    if frame_count % 30 == 0:
        print(f"Processed frame {frame_count}/{total_frames} - Detections: {detections}")
    
    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopped by user")
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nProcessing complete! Processed {frame_count} frames")
