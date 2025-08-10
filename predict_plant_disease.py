from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json

# Load the trained model
model = load_model('plant_disease_model.keras')

# Load class labels (names of diseases)
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
labels = {v: k for k, v in class_indices.items()}

# Load and prepare the new image
img_path = 'predict.JPG'  # put your new image path here
img = image.load_img(img_path, target_size=(160, 160))  # same size you trained on
x = image.img_to_array(img) / 255.0  # scale pixels like during training
x = np.expand_dims(x, axis=0)  # add batch dimension

# Predict
pred = model.predict(x)
predicted_class = np.argmax(pred, axis=1)[0]

print("Predicted disease:", labels[predicted_class])
