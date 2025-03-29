import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_excel("Dataset.xlsx", sheet_name="Dataset")

# Selecting features and target variable
features = ['person_age', 'person_income']  # Selecting two numerical features for simplicity
target = 'loan_status'
X = df[features]
y = df[target]

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train kNN model (k=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)

# A1: Confusion Matrix and Performance Metrics
conf_matrix = confusion_matrix(y_test, y_pred_test)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_report(y_test, y_pred_test))

# A2: Regression Metrics (assuming a previous price prediction task)
y_actual = np.random.rand(len(y_test)) * 100  # Placeholder values
y_pred = y_actual + np.random.normal(0, 10, len(y_test))

mse = mean_squared_error(y_actual, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
r2 = r2_score(y_actual, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("MAPE:", mape)
print("R2 Score:", r2)

# A3: Generate and visualize training data for kNN
np.random.seed(42)
X_train_knn = np.random.uniform(1, 10, (20, 2))
y_train_knn = np.random.choice([0, 1], 20)

plt.figure(figsize=(8, 6))
plt.scatter(X_train_knn[:, 0], X_train_knn[:, 1], c=y_train_knn, cmap='bwr', edgecolors='k')
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.title("Training Data")
plt.show()

# A4: Generate and classify test data using kNN
test_points = np.array([[x, y] for x in np.arange(0, 10, 0.1) for y in np.arange(0, 10, 0.1)])
preds = knn.predict(test_points)

plt.figure(figsize=(8, 6))
plt.scatter(test_points[:, 0], test_points[:, 1], c=preds, cmap='bwr', alpha=0.5, edgecolors='k')
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.title("Test Data Classification")
plt.show()

# A5: Repeat A4 for multiple k values
for k in [1, 5, 10]:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train_knn, y_train_knn)
    preds_k = knn_k.predict(test_points)
    plt.figure(figsize=(8, 6))
    plt.scatter(test_points[:, 0], test_points[:, 1], c=preds_k, cmap='bwr', alpha=0.5, edgecolors='k')
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.title(f"Test Data Classification (k={k})")
    plt.show()

# A6: Repeat for real project data
knn_real = KNeighborsClassifier(n_neighbors=3)
knn_real.fit(X_train, y_train)
y_project_pred = knn_real.predict(X_test)
print("Project Data Classification Report:\n", classification_report(y_test, y_project_pred))

# A7: Hyperparameter tuning
param_grid = {'n_neighbors': range(1, 20)}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("Best k value:", grid_search.best_params_['n_neighbors'])
