from collections import defaultdict
import supervision as sv
from ultralytics import YOLO
import cv2

def detect_vehicles(video_path, output_path="output/output_multi_gate_vehicle_detection.mp4"):
    # Load the YOLOv8 model
    model = YOLO('yolov8x.pt')

    # Set up video capture
    cap = cv2.VideoCapture(video_path)

    # Define the line coordinates up to 8 gates
    START_1, END_1 = sv.Point(10, 135), sv.Point(87, 151)
    START_2, END_2 = sv.Point(87, 151), sv.Point(152, 166)
    START_3, END_3 = sv.Point(152, 166), sv.Point(219, 180)
    START_4, END_4 = sv.Point(219, 180), sv.Point(287, 195)
    START_5, END_5 = sv.Point(287, 195), sv.Point(356, 210)
    START_6, END_6 = sv.Point(356, 210), sv.Point(428, 226)
    START_7, END_7 = sv.Point(390, 245), sv.Point(486, 267)
    START_8, END_8 = sv.Point(486, 267), sv.Point(562, 284)

     # Define the gates
    gates = [
        (START_1, END_1),
        (START_2, END_2),
        (START_3, END_3),
        (START_4, END_4),
        (START_5, END_5),
        (START_6, END_6),
        (START_7, END_7),
        (START_8, END_8)
    ]

    # Store the track history
    track_history = defaultdict(lambda: [])

    # Create dictionaries to keep track of objects that have crossed the lines
    crossed_objects = [defaultdict(bool) for _ in range(8)]

    # Counters for cars and buses
    car_count = 0
    bus_count = 0

    # Open a video sink for the output video
    video_info = sv.VideoInfo.from_video_path(video_path)
    with sv.VideoSink(output_path, video_info) as sink:
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                results = model.track(frame, classes=[2, 5], conf=0.15, persist=True, tracker="bytetrack.yaml")

                # Get the boxes and track IDs
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()

                # Visualize the results on the frame
                annotated_frame = results[0].plot()
                detections = sv.Detections.from_ultralytics(results[0])

                # Process each detection
                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 30 tracks for 30 frames
                        track.pop(0)

                    for i, (start, end) in enumerate(gates):
                        # Check if the object crosses the line
                        if start.x < x < end.x and start.y < y < end.y:
                            # Mark the object as crossed
                            if not crossed_objects[i][track_id]:
                                crossed_objects[i][track_id] = True

                                # add the counters based on the class ID
                                if class_id == 2:  # Car
                                    car_count += 1
                                elif class_id == 5:  # Bus
                                    bus_count += 1

                            # Annotate the object as it crosses the line
                            cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)),
                                          (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)

                # Draw the lines on the frame
                for start, end in gates:
                    cv2.line(annotated_frame, (start.x, start.y), (end.x, end.y), (0, 255, 0), 2)

                # Label the gates
                for i, xpos in enumerate([50, 115, 180, 250, 315, 380, 460, 535]):
                    gate_text = f"{i}"
                    cv2.putText(annotated_frame, gate_text, (xpos, int((60771 + 125*xpos)/569)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                # Write the count of objects on each frame
                for i, ypos in enumerate(range(20, 100, 10)):
                    count_text = f"Objects crossed gate {i+1}: {len(crossed_objects[i])}"
                    cv2.putText(annotated_frame, count_text, (10, ypos), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    
                # Display the vehicle counts on the frame
                cv2.putText(annotated_frame, f"Cars: {car_count}", (310, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(annotated_frame, f"Buses: {bus_count}", (310, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

                # Write the frame with annotations to the output video
                sink.write_frame(annotated_frame)
            else:
                break

    # Release the video capture
    cap.release()

if __name__ == "__main__":
    video_path = "data/toll_gate.mp4"  # Replace with your video path
    detect_vehicles(video_path)