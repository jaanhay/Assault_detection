import xml.etree.ElementTree as ET
import pandas as pd
import os


def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    filename = root.find('filename').text

    data = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        data.append([filename,name, xmin, ymin, xmax, ymax])
    return data


def convert_annotations_to_csv(xml_folder, output_csv):
    all_data = []
    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith('.xml'):
            file_path = os.path.join(xml_folder, xml_file)
            data = parse_xml(file_path)
            all_data.extend(data)

    df = pd.DataFrame(all_data, columns=['filename','class', 'xmin', 'ymin', 'xmax', 'ymax'])
    df.to_csv(output_csv, index=False)


# Convert XML annotations
convert_annotations_to_csv('Violence-Image-Dataset-master/skeleton/xml', 'annotations.csv')



from sklearn.model_selection import train_test_split
import shutil
import os


def prepare_dataset(image_folder, annotations_csv, output_folder):
    df = pd.read_csv(annotations_csv)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['class'])

    # Create folders
    os.makedirs(os.path.join(output_folder, 'train', 'normal'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'train', 'fighting'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'val', 'normal'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'val', 'fighting'), exist_ok=True)

    def copy_images(df, folder):
        for _, row in df.iterrows():
            class_folder = row['class']
            if class_folder not in ['normal', 'fighting']:
                print(f"Skipping unknown class: {class_folder}")
                continue

            src = os.path.join(image_folder, row['filename'])
            dst = os.path.join(output_folder, folder, class_folder, row['filename'])

            if os.path.exists(src):
                shutil.copy(src, dst)
                print(f"Copied {src} to {dst}")
            else:
                print(f"Source file {src} does not exist.")

    print("Preparing training images...")
    copy_images(train_df, 'train')

    print("Preparing validation images...")
    copy_images(val_df, 'val')


# Call the function with your paths
prepare_dataset('Violence-Image-Dataset-master/skeleton/images', 'annotations.csv', 'dataset')


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Define the data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    'dataset/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Define the model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10, validation_data=val_generator)
model.save('my_model.h5')
