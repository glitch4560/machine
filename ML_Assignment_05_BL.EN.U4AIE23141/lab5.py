import numpy as np
import zipfile
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Extract the dataset
dataset_path = "C:\AIO\Semster Files\SEMSTER - 4\ML\Lab Work\ML_Assignment_05_BL.EN.U4AIE23138\Dataset.zip"
extract_path = "C:\AIO\Semster Files\SEMSTER - 4\ML\Lab Work\ML_Assignment_05_BL.EN.U4AIE23138\Dataset"
with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Load MRI images and preprocess
image_size = (128, 128)
X = []
y = []
for idx, filename in enumerate(sorted(os.listdir(extract_path))):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(extract_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, image_size).flatten()  # Convert to 1D array
        X.append(img)
        y.append(idx % 2)  # Dummy labels (0 or 1)

X = np.array(X)
y = np.array(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- A1: Train Linear Regression on One Attribute ---
X_train_single = X_train[:, [0]]
X_test_single = X_test[:, [0]]

reg = LinearRegression().fit(X_train_single, y_train)
y_train_pred = reg.predict(X_train_single)
y_test_pred = reg.predict(X_test_single)

# --- A2: Evaluate Model Performance ---
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_train_pred)

mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_test_pred)

print("A2 - Model Evaluation:")
print(f"Train MSE: {mse_train}, RMSE: {rmse_train}, R2: {r2_train}")
print(f"Test MSE: {mse_test}, RMSE: {rmse_test}, R2: {r2_test}")

# --- A3: Train Linear Regression on All Attributes ---
reg_all = LinearRegression().fit(X_train, y_train)
y_train_pred_all = reg_all.predict(X_train)
y_test_pred_all = reg_all.predict(X_test)

mse_train_all = mean_squared_error(y_train, y_train_pred_all)
rmse_train_all = np.sqrt(mse_train_all)
r2_train_all = r2_score(y_train, y_train_pred_all)

mse_test_all = mean_squared_error(y_test, y_test_pred_all)
rmse_test_all = np.sqrt(mse_test_all)
r2_test_all = r2_score(y_test, y_test_pred_all)

print("A3 - Model Evaluation on All Features:")
print(f"Train MSE: {mse_train_all}, RMSE: {rmse_train_all}, R2: {r2_train_all}")
print(f"Test MSE: {mse_test_all}, RMSE: {rmse_test_all}, R2: {r2_test_all}")

# --- A4: Perform K-Means Clustering ---
kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X_train)
cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

print("A4 - K-Means Clustering Done")

# --- A5: Compute Clustering Metrics ---
silhouette = silhouette_score(X_train, kmeans.labels_)
calinski_harabasz = calinski_harabasz_score(X_train, kmeans.labels_)
davies_bouldin = davies_bouldin_score(X_train, kmeans.labels_)

print("A5 - Clustering Evaluation Metrics:")
print(f"Silhouette Score: {silhouette}")
print(f"Calinski-Harabasz Score: {calinski_harabasz}")
print(f"Davies-Bouldin Index: {davies_bouldin}")

# --- A6: K-Means for Different K Values ---
k_values = range(2, min(10, len(X_train)))  # Avoid k > number of samples
silhouette_scores = []
calinski_harabasz_scores = []
davies_bouldin_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_train)
    silhouette_scores.append(silhouette_score(X_train, kmeans.labels_))
    calinski_harabasz_scores.append(calinski_harabasz_score(X_train, kmeans.labels_))
    davies_bouldin_scores.append(davies_bouldin_score(X_train, kmeans.labels_))

# Plot scores vs k values
plt.figure(figsize=(10, 4))
plt.plot(k_values, silhouette_scores, label="Silhouette Score")
plt.plot(k_values, calinski_harabasz_scores, label="Calinski-Harabasz Score")
plt.plot(k_values, davies_bouldin_scores, label="Davies-Bouldin Index")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Score")
plt.title("Clustering Evaluation Metrics vs k")
plt.legend()
plt.show()

# --- A7: Elbow Method for Optimal k ---
distortions = []
for k in range(2, min(10, len(X_train))):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_train)
    distortions.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(2, min(10, len(X_train))), distortions, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.show()
