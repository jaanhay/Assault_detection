from ultralytics import YOLO

import tensorflow as tf
import cv2
# Load the trained model
model = tf.keras.models.load_model('my_model.h5')


model1 = YOLO('yolov8m-pose.pt')

import numpy as np

import numpy as np
def extract_pose_data(result):
    poses = []
    for pose in result.xyxy[0].itertuples():
        # Extract keypoints if availableqqq
        keypoints = pose.keypoints if hasattr(pose, 'keypoints') else []
        poses.append(keypoints)
    return poses

def preprocess_pose_data(keypoints, expected_length=51):
    processed_data = np.array(keypoints).flatten()

    # Check if the processed data length matches the expected length
    if len(processed_data) < expected_length:
        # If shorter, pad with zeros
        processed_data = np.pad(processed_data, (0, expected_length - len(processed_data)), 'constant')
    elif len(processed_data) > expected_length:
        # If longer, truncate
        processed_data = processed_data[:expected_length]

    # Normalize
    processed_data = processed_data / 255.0  # Adjust if your model expects different normalization
    # Add batch dimension
    return np.expand_dims(processed_data, axis=0)


results = model1(source=0, conf=0.5, show=True)
print("hellooooooo")
for result in results:

    poses = extract_pose_data(result)
    for pose in poses:
            # Preprocess pose data
        processed_pose = preprocess_pose_data(pose)

            # Predict using the `.h5` model
        prediction = model.predict(processed_pose)

            # If prediction indicates assault (adjust threshold as needed)
        if prediction[0] > 0.4:  # Adjust threshold based on your model's output

            bbox = pose.bbox
            result.plot(show=True, boxes=[(*bbox, 'red')])
            print("assault")
        else:
            print("normal")


        # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
