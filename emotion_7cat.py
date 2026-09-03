import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

%cd Face-Emotion-Recognition

# 1. Path setup
data_dir = 'data/BTFER7'
model_path = 'models/model_batch7.h5'

# 2. Automatically load and sort the 8 category folder names
categories = sorted(
    [
        d
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')
    ]
)
print(f'Loaded Categories ({len(categories)}): {categories}')

# 3. Load your updated model
model = load_model(model_path)


# 4. Helper function to preprocess & predict on an image
def predict_emotion(image_path):
  # Open and resize image matching model input (64x64x3)
  img = Image.open(image_path).convert('RGB')
  img_resized = img.resize((64, 64))

  # Convert to array and normalize (0.0 to 1.0)
  img_array = np.array(img_resized) / 255.0
  img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 64, 64, 3)

  # Predict
  predictions = model.predict(img_array)[0]
  predicted_class_idx = np.argmax(predictions)
  predicted_label = categories[predicted_class_idx]
  confidence = predictions[predicted_class_idx] * 100

  return predicted_label, confidence, predictions
