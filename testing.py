from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the model
model = load_model('my_model.h5')

# Prepare the test dataset (modify path and parameters as needed)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'dataset/val',  # directory containing test data
    target_size=(224, 224),  # size to which images will be resized
    batch_size=32,
    class_mode='binary'  # or 'binary' depending on your problem
)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_generator)

print(f'Test Accuracy: {accuracy * 100:.2f}%')
