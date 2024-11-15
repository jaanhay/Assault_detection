import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf

# Load YOLOv8 model for pose estimation
model = YOLO('yolov8m-pose.pt')

# Load assault detection model
assault_model = tf.keras.models.load_model('my_model.h5')

# Load class list
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('videoplayback.mp4')


def extract_pose_data(result):
    poses = []
    if hasattr(result, 'keypoints') and result.keypoints is not None:
        for pose in result.keypoints:
            keypoints = pose.xy[0]  # Extract keypoints
            poses.append(keypoints)
    return poses


def preprocess_pose_data(keypoints, expected_length=51):
    processed_data = np.array(keypoints).flatten()
    if len(processed_data) < expected_length:
        processed_data = np.pad(processed_data, (0, expected_length - len(processed_data)), 'constant')
    elif len(processed_data) > expected_length:
        processed_data = processed_data[:expected_length]
    return np.expand_dims(processed_data / 255.0, axis=0)


def draw_boxes(frame, results, class_list):
    person_count = 0
    for result in results:
        for detection in result.boxes:
            class_id = int(detection.cls[0])
            class_name = result.names[class_id]

            if class_name == 'person':  # Check specifically for 'person' class
                x1, y1, x2, y2 = map(int, detection.xyxy[0])
                color = (0, 255, 0)
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                person_count += 1
                label = f"{class_name} {person_count}"
                frame = cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                poses = extract_pose_data(result)
                for pose in poses:
                    processed_pose = preprocess_pose_data(pose)
                    prediction = assault_model.predict(processed_pose)

                    if prediction[0] > 0.4:  # Adjust threshold based on your model's output
                        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box for assault
                        frame = cv2.putText(frame, "Assault Detected", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            (0, 0, 255), 2)
                    else:
                        frame = cv2.putText(frame, "Normal Behavior", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            (0, 255, 0), 2)

    return frame


count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(source=frame, conf=0.5)
    frame = draw_boxes(frame, results, class_list)
    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
