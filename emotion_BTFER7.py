import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

# Image dimensions
img_height = 64
img_width = 64

# Folder setup
data_dir = 'data/BTFER7'

# Filter subdirectories and avoid system files like .DS_Store
categories = sorted(
    [
        d
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')
    ]
)
num_classes = len(categories)
print(f'Detected {num_classes} classes: {categories}')


# Load images
def load_all_images(base_dir):
  data, labels = [], []
  for idx, category in enumerate(categories):
    category_path = os.path.join(base_dir, category)
    for img_name in os.listdir(category_path):
      if img_name.startswith('.'):
        continue
      img_path = os.path.join(category_path, img_name)
      try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((img_width, img_height))
        data.append(np.array(img))
        labels.append(idx)
      except Exception as e:
        print(f'Error loading image {img_path}: {e}')
  return np.array(data), np.array(labels)


# Load dataset and normalize
X, y = load_all_images(data_dir)
X = X / 255.0

# Train/Test Split (80/20 ratio)
X_train, X_test, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# One-hot encode targets
y_train = to_categorical(y_train_raw, num_classes=num_classes)
y_test = to_categorical(y_test_raw, num_classes=num_classes)

# Callbacks & Optimizer
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1
)
early_stopping = EarlyStopping(
    monitor='val_loss', patience=7, restore_best_weights=True, verbose=1
)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Model architecture for 8 classes
model = Sequential([
    Conv2D(
        128,
        (3, 3),
        activation='relu',
        kernel_regularizer=l2(0.001),
        input_shape=(img_height, img_width, 3),
    ),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (2, 2), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.0005)),
    Dropout(0.5),
    Dense(num_classes, activation='softmax'),
])

model.compile(
    optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy']
)

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[reduce_lr, early_stopping],
)

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Predictions & Evaluation
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=categories,
    yticklabels=categories,
)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Classification Report
print('Classification Report:')
print(classification_report(y_true, y_pred_classes, target_names=categories))

# Save output explicitly in .h5 format
save_dir = 'models'
os.makedirs(save_dir, exist_ok=True)
model.save(os.path.join(save_dir, 'model_batch7.h5'))
print("Model saved successfully in 'models/model_batch7.h5'")
