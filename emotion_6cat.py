import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
%cd Face-Emotion-Recognition

# Load your saved model
model = load_model('models/model2.h5')

# Categories matched to your training setup (sorted order)
categories = ['angry', 'disgusted', 'neutral', 'sad', 'smiling', 'surprise']
