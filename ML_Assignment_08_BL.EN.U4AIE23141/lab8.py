import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# === Dataset Path ===
DATASET_PATH = "C:\\AIO\\Semster Files\\SEMSTER - 4\\ML\\Lab Work\\ML_Assignment_08_BL.EN.U4AIE23138\\Dataset"

# === Load Dataset ===
def load_glioma_dataset():
    data = []
    for fname in os.listdir(DATASET_PATH):
        img_path = os.path.join(DATASET_PATH, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64)).flatten()
            data.append(img)
    return np.array(data)

# === A1: Core Functions ===
def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)
def relu(x): return np.maximum(0, x)
def leaky_relu(x): return np.where(x > 0, x, 0.01 * x)

def a1_demo(X_sample):
    print("A1: Activations on first image vector:")
    print("Sigmoid:", sigmoid(X_sample[:5]))
    print("Tanh:", tanh(X_sample[:5]))
    print("ReLU:", relu(X_sample[:5]))
    print("Leaky ReLU:", leaky_relu(X_sample[:5]))

# === A2-A3: Perceptron ===
def train_perceptron(X, y, weights, lr=0.01, max_epochs=1000, threshold=0.01):
    errors = []
    for epoch in range(max_epochs):
        total_error = 0
        for i in range(len(X)):
            pred = 1 if np.dot(weights, X[i]) >= 0 else 0
            error = y[i] - pred
            total_error += error ** 2
            weights += lr * error * X[i]
        errors.append(total_error)
        if total_error <= threshold:
            break
    return weights, errors

# === A4: Learning Rate Plot ===
def a4_learning_rate_plot(X, y):
    rates = np.arange(0.01, 0.11, 0.01)
    epochs_list = []
    for lr in rates:
        _, errors = train_perceptron(X, y.copy(), np.random.rand(X.shape[1]), lr)
        epochs_list.append(len(errors))
    plt.plot(rates, epochs_list, marker='o')
    plt.title("A4: Learning Rate vs Epochs")
    plt.xlabel("Learning Rate")
    plt.ylabel("Epochs")
    plt.grid(True)
    plt.show()

# === A5: Repeat with XOR-Like Labels ===
def xor_like_labels(X):
    return np.logical_xor(X[:, 0] > 127, X[:, 1] > 127).astype(int)

# === A6: Perceptron with Sigmoid Activation ===
def a6_sigmoid_train(X, y, alpha=0.01, max_epochs=1000):
    w = np.random.randn(X.shape[1])
    for _ in range(max_epochs):
        for xi, yi in zip(X, y):
            pred = sigmoid(np.dot(xi, w))
            error = yi - (pred >= 0.5)
            w += alpha * error * xi
    return w

# === A7: Pseudo-Inverse ===
def a7_pseudo_inverse(X, y):
    print("A7: Pseudo-inverse solution shape:", np.linalg.pinv(X).shape)
    return np.linalg.pinv(X) @ y

# === A8-A9: Backpropagation ===
def a8_backprop(X, y, alpha=0.01, max_epochs=500):
    input_dim, hidden_dim = X.shape[1], 10
    w1 = np.random.randn(input_dim, hidden_dim)
    w2 = np.random.randn(hidden_dim, 1)
    errors = []
    for _ in range(max_epochs):
        h = sigmoid(X @ w1)
        out = sigmoid(h @ w2).flatten()
        err = y - out
        errors.append(np.sum(err ** 2))
        if errors[-1] <= 0.01:
            break
        delta_out = (err * out * (1 - out))[:, None]
        delta_hid = delta_out @ w2.T * h * (1 - h)
        w2 += alpha * h.T @ delta_out
        w1 += alpha * X.T @ delta_hid
    return errors

# === A10: Multi-output Mapping ===
def a10_multi_output(y):
    return np.array([[0, 1] for _ in y])

# === A11: sklearn MLPClassifier ===
def a11_mlp_classifier(X, y):
    model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42)
    model.fit(X, y)
    print("A11: Sklearn MLPClassifier Accuracy:", model.score(X, y))

# === A12: Final Evaluation ===
def a12_final(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("A12: Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# === Run All ===
X = load_glioma_dataset()
y = np.zeros(len(X))  # Only one class

a1_demo(X[0])                            # A1
_, errors_a2 = train_perceptron(X, y, np.random.rand(X.shape[1]))  # A2-A3
a4_learning_rate_plot(X, y)             # A4
xor_labels = xor_like_labels(X)         # A5
train_perceptron(X, xor_labels, np.random.rand(X.shape[1]))        # A5 continued
a6_sigmoid_train(X, y)                  # A6
a7_pseudo_inverse(X, y)                 # A7
backprop_errors = a8_backprop(X, y)     # A8-A9
plt.plot(backprop_errors)
plt.title("A9: Backprop Error Curve")
plt.xlabel("Epoch")
plt.ylabel("SSE")
plt.grid(True)
plt.show()
print("A10:", a10_multi_output(y)[:3])  # A10
a11_mlp_classifier(X, y)                # A11
a12_final(X, y)                         # A12
