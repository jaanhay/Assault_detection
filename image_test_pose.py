import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

model = tf.keras.models.load_model('my_model.h5')
model1 = YOLO('yolov8m-pose.pt')

def preprocess_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = image_array / 255.0
    return np.expand_dims(image_array, axis=0)

def process_and_display_image(image_path, x, y, w, h, threshold=0.3):
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    image_cv = cv2.imread(image_path)
    if prediction[0] > threshold:
        print("fighting detected")
        cv2.rectangle(image_cv, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow(f'Prediction Result - {image_path}', image_cv)
    cv2.waitKey(0)

image_paths = ['mob2.jpg', 'mobile_webp.webp']
bounding_boxes = [(30, 30, 150, 150), (30, 30, 150, 150)]

for image_path, bbox in zip(image_paths, bounding_boxes):
    process_and_display_image(image_path, *bbox)

cv2.destroyAllWindows()
