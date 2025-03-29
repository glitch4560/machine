import os
import numpy as np
import cv2
import zipfile
import math
import matplotlib.pyplot as plt
from skimage.feature import hog
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import DecisionBoundaryDisplay

# Debugging function
def debug_print(msg, var=None):
    print(f"{msg}")
    if var is not None:
        print(var)

# Step 1: Unzip & Load Dataset
dataset_path = "C:\\AIO\\Semster Files\\SEMSTER - 4\\ML\\Lab Work\\ML_Assignment_06_BL.EN.U4AIE23138\\Dataset.zip"
extract_path = "C:\\AIO\\Semster Files\\SEMSTER - 4\\ML\\Lab Work\\ML_Assignment_06_BL.EN.U4AIE23138\\Dataset"

with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

debug_print("Dataset extracted successfully.")

# Step 2: Feature Extraction using HOG
def extract_hog_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        debug_print(f"Error: Could not read image {image_path}")
        return None  # Skip unreadable images
    image = cv2.resize(image, (128, 128))  
    features, _ = hog(image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
    return features

# Load all images
image_files = [os.path.join(extract_path, f) for f in os.listdir(extract_path) if f.endswith(".jpg")]
debug_print("Total images found:", len(image_files))

# Assign Labels
labels = ["Tumor" if "gl" in os.path.basename(f) else "No-Tumor" for f in image_files]
debug_print("Sample labels:", labels[:5])

# Encode Labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Extract HOG Features
X = np.array([extract_hog_features(f) for f in image_files if extract_hog_features(f) is not None])

# Ensure X and y have the same length
if len(X) != len(y):
    debug_print("Error: Mismatch in features and labels!", (len(X), len(y)))
    exit()

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

debug_print("Dataset Loaded and Features Extracted!")
debug_print(f"Number of Training Samples: {len(y_train)}")
debug_print(f"Classes in Training Set: {Counter(y_train)}")

# ------------------------------------------
# A1: Calculate Entropy
# ------------------------------------------
def entropy(y):
    counts = Counter(y)
    total = len(y)
    if total == 0:
        return 0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())

train_entropy = entropy(y_train)
debug_print("Entropy of Dataset:", train_entropy)

# ------------------------------------------
# A2: Calculate Gini Index
# ------------------------------------------
def gini_index(y):
    counts = Counter(y)
    total = len(y)
    return 1 - sum((count / total) ** 2 for count in counts.values())

train_gini = gini_index(y_train)
debug_print("Gini Index of Dataset:", train_gini)

# ------------------------------------------
# A3 & A4: Identify Root Node using Information Gain
# ------------------------------------------
def information_gain(X, y, feature_index, bins=10):
    """Calculate Information Gain for a given feature."""
    total_entropy = entropy(y)

    # Binning Continuous Features
    feature_values = X[:, feature_index]
    
    if len(np.unique(feature_values)) == 1:  # If all feature values are the same
        return 0

    bin_edges = np.linspace(feature_values.min(), feature_values.max(), bins + 1)
    binned_feature = np.digitize(feature_values, bin_edges)

    # Calculate Entropy for each split
    split_entropy = 0
    for value in np.unique(binned_feature):
        subset_y = y[binned_feature == value]
        if len(subset_y) == 0:
            continue
        split_entropy += (len(subset_y) / len(y)) * entropy(subset_y)

    return total_entropy - split_entropy

# Find best feature
best_feature = max(range(X_train.shape[1]), key=lambda i: information_gain(X_train, y_train, i))
debug_print(f"Best Root Node Feature: Feature {best_feature}")

# ------------------------------------------
# A5: Build & Train Decision Tree
# ------------------------------------------
dt_classifier = DecisionTreeClassifier(criterion="entropy", random_state=42)
dt_classifier.fit(X_train, y_train)

# Test Accuracy
y_pred = dt_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
debug_print("Decision Tree Accuracy:", accuracy)

# ------------------------------------------
# A6: Visualize Decision Tree
# ------------------------------------------
plt.figure(figsize=(12, 8))
tree.plot_tree(dt_classifier, feature_names=[f"Feature {i}" for i in range(X_train.shape[1])], class_names=["No-Tumor", "Tumor"], filled=True)
plt.title("Decision Tree Visualization")
plt.show()

# Debugging: Check if tree is correctly built
debug_print("Decision Tree Structure:", dt_classifier.tree_)

# ------------------------------------------
# A7: Plot Decision Boundaries (Using 2 Features)
# ------------------------------------------
if X_train.shape[1] >= 2:  # Ensure at least 2 features exist
    X_plot = X_train[:, :2]  # Take first 2 features
    y_plot = y_train

    dt_2d = DecisionTreeClassifier(criterion="entropy", random_state=42)
    dt_2d.fit(X_plot, y_plot)

    plt.figure(figsize=(8, 6))
    DecisionBoundaryDisplay.from_estimator(dt_2d, X_plot, response_method="predict", cmap=plt.cm.Paired, alpha=0.5)
    plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot, edgecolors="k", cmap=plt.cm.Paired)
    plt.title("Decision Boundary of Decision Tree (Using 2 Features)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
else:
    debug_print("Error: Not enough features to plot decision boundaries.")
