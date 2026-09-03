import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

# Set image dimensions
img_height = 64
img_width = 64

# Updated directories
train_dir = 'data/6_CategoryProcessed/train'
test_dir = 'data/6_CategoryProcessed/test'

# Load categories from train_dir (same for test)
categories = sorted(os.listdir(train_dir))

# Function to load images
def load_images(base_dir):
    data, labels = [], []
    for idx, category in enumerate(categories):
        category_path = os.path.join(base_dir, category)
        if not os.path.isdir(category_path):
            continue
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((img_width, img_height))
                data.append(np.array(img))
                labels.append(idx)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return np.array(data), np.array(labels)

# Load and normalize data
X_train_raw, y_train_raw = load_images(train_dir)
X_test_raw, y_test_raw = load_images(test_dir)

X_train_raw = X_train_raw / 255.0
X_test_raw = X_test_raw / 255.0

# Apply SMOTE only on training data
data_flat = X_train_raw.reshape(X_train_raw.shape[0], -1)
smote = SMOTE(random_state=42)
data_resampled, labels_resampled = smote.fit_resample(data_flat, y_train_raw)
data_resampled = data_resampled.reshape(-1, img_height, img_width, 3)

# One-hot encode
labels_resampled = to_categorical(labels_resampled, num_classes=len(categories))
y_test = to_categorical(y_test_raw, num_classes=len(categories))

# Shuffle training data
indices = np.arange(data_resampled.shape[0])
np.random.shuffle(indices)
X_train = data_resampled[indices]
y_train = labels_resampled[indices]
X_test = X_test_raw
y_test = y_test

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Model architecture
model = Sequential([
    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001), input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (2, 2), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.0005)),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')
])

# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[reduce_lr, early_stopping]
)

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=categories))

# Save model
save_dir = 'models'
os.makedirs(save_dir, exist_ok=True)
model.save(os.path.join(save_dir, 'model2.h5'))
print("Model saved successfully in 'models/model2.h5'")