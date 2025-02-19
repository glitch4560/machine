import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Load dataset
file_path = "Dataset.xlsx"
df = pd.read_excel(file_path, sheet_name='Dataset')

# Select features and class labels
features = ['person_income', 'loan_amnt']
X = df[features].values
y = df['loan_status'].values

# A1: Intraclass Spread and Interclass Distance
class_0 = X[y == 0]
class_1 = X[y == 1]

centroid_0 = np.mean(class_0, axis=0)
centroid_1 = np.mean(class_1, axis=0)

spread_0 = np.std(class_0, axis=0)
spread_1 = np.std(class_1, axis=0)

interclass_distance = np.linalg.norm(centroid_0 - centroid_1)
print("Centroid 0:", centroid_0)
print("Centroid 1:", centroid_1)
print("Spread 0:", spread_0)
print("Spread 1:", spread_1)
print("Interclass Distance:", interclass_distance)

# A2: Histogram for 'person_income'
plt.hist(df['person_income'], bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('Person Income')
plt.ylabel('Frequency')
plt.title('Histogram of Person Income')
plt.show()

mean_income = np.mean(df['person_income'])
var_income = np.var(df['person_income'])
print("Mean Income:", mean_income)
print("Variance of Income:", var_income)

# A3: Minkowski Distance for Two Feature Vectors
vec1, vec2 = X[0], X[1]
minkowski_distances = [np.linalg.norm(vec1 - vec2, ord=r) for r in range(1, 11)]
plt.plot(range(1, 11), minkowski_distances, marker='o')
plt.xlabel('Order r')
plt.ylabel('Minkowski Distance')
plt.title('Minkowski Distance Variation')
plt.show()

# A4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# A5: Train k-NN Classifier (k=3)
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

# A6: Evaluate Accuracy
accuracy = neigh.score(X_test, y_test)
print("k-NN Accuracy:", accuracy)

# A7: Prediction Behavior
predictions = neigh.predict(X_test)
print("Predictions:", predictions[:10])

# A8: Vary k from 1 to 11
k_values = range(1, 12)
train_accuracies = []
test_accuracies = []
k_nn_accuracies = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_accuracies.append(knn.score(X_train, y_train))
    test_accuracies.append(knn.score(X_test, y_test))
    k_nn_accuracies.append(knn.score(X_test, y_test))

plt.plot(k_values, k_nn_accuracies, marker='o', label='k-NN Accuracy')
plt.xlabel('k Value')
plt.ylabel('Accuracy')
plt.title('k-NN Accuracy vs k Value')
plt.legend()
plt.show()

plt.plot(k_values, train_accuracies, marker='o', label='Train Accuracy')
plt.xlabel('k Value')
plt.ylabel('Accuracy')
plt.title('k vs Train Accuracy')
plt.legend()
plt.show()

plt.plot(k_values, test_accuracies, marker='o', label='Test Accuracy')
plt.xlabel('k Value')
plt.ylabel('Accuracy')
plt.title('k vs Test Accuracy')
plt.legend()
plt.show()

# A9: Confusion Matrix and Classification Report
y_pred = neigh.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)