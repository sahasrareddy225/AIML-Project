# ==================================================
# train_model.py
# ==================================================
# 1️⃣ Dataset type
#  - Labelled, balanced/unbalanced check
#  - CSV data containing pixel values of hand sign images
# ==================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import joblib

# ==================================================
# 2️⃣ Load dataset
# ==================================================
train_df = pd.read_csv("data/sign_mnist_train.csv")
test_df = pd.read_csv("data/sign_mnist_test.csv")

# Separate features and labels
y_train = train_df['label'].values
x_train = train_df.drop(['label'], axis=1).values
y_test = test_df['label'].values
x_test = test_df.drop(['label'], axis=1).values

print(f"Dataset loaded ✅\nTrain shape: {x_train.shape}, Test shape: {x_test.shape}")

# ==================================================
# 3️⃣ Check data balance
# ==================================================
sns.countplot(x=y_train)
plt.title("Training Data Distribution")
plt.show()

# ==================================================
# 4️⃣ Pre-processing
# ==================================================
# Normalize pixel values (0–255 → 0–1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape to 28x28 images (grayscale)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# ==================================================
# 5️⃣ Balancing technique
# (We’ll use ImageDataGenerator for data augmentation)
# ==================================================
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(x_train)

# ==================================================
# 6️⃣ Cross-validation setup
# ==================================================
print("Performing simple train/test split (80/20)...")

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

# ==================================================
# 7️⃣ Feature extraction / selection
# For images, CNN automatically extracts features.
# ==================================================

# ==================================================
# 8️⃣ Encode labels (one-hot)
# ==================================================
num_classes = y_train.max() + 1  # ensures last label included
print(f"✅ Number of detected classes: {num_classes}")

y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_val_cat = to_categorical(y_val, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

# ==================================================
# 9️⃣ CNN Model Architecture
# ==================================================
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# ==================================================
#  🔟 Optimization technique
# ==================================================
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# ==================================================
# 11️⃣ Train model
# ==================================================
history = model.fit(
    datagen.flow(x_train, y_train_cat, batch_size=64),
    validation_data=(x_val, y_val_cat),
    epochs=10,
    verbose=1
)

# ==================================================
# 12️⃣ Evaluate model
# ==================================================
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"✅ Test Accuracy: {test_acc:.4f}")

# ==================================================
# 13️⃣ Performance metrics and plots
# ==================================================
y_pred = np.argmax(model.predict(x_test), axis=1)
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Training curves
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend(); plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(); plt.title("Loss")
plt.show()

# ==================================================
# 14️⃣ Save model and label map
# ==================================================
model.save("sign_model.keras")
label_map = {i: chr(65 + i) for i in range(num_classes)}  # A–Z
joblib.dump(label_map, "label_map.pkl")
print("✅ Model and label map saved successfully.")
