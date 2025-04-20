import os
import numpy as np
import cv2
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM

# Path to dataset folder containing glioma images
DATASET_PATH ="C:\\AIO\\Semster Files\\SEMSTER - 4\\ML\\Lab Work\\ML_Assignment_07_BL.EN.U4AIE23138\\Dataset"

# Load images from the dataset
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)  # Read image
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Resize all images to (128x128)
            images.append(img)
    return np.array(images)

# Load the glioma images
X = load_images_from_folder(DATASET_PATH)
y = np.zeros(len(X), dtype=int)  # Assign all labels as 0 (glioma class)

# Check class distribution
print("Class distribution before split:", Counter(y))

# Handle the case where only one class is present
unique_classes = np.unique(y)
if len(unique_classes) < 2:
    print("Only one class detected. Proceeding with one-class classification.")

# Split the dataset (without stratification)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

# Train a One-Class SVM model for anomaly detection
model = OneClassSVM(gamma='auto').fit(X_train.reshape(len(X_train), -1))

# Predict on test data
predictions = model.predict(X_test.reshape(len(X_test), -1))

# Output results
print("Predictions:", predictions)
print("Note: In One-Class SVM, +1 means 'normal' (glioma), -1 means 'outliers' (anomalies).")
