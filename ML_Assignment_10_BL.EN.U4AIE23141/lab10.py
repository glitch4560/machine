import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime.lime_tabular

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# === Dataset Path ===
DATASET_PATH = "C:\\AIO\\Semster Files\\SEMSTER - 4\\ML\\Lab Work\\ML_Assignment_09_BL.EN.U4AIE23138\\Dataset"

# === Load Glioma Dataset ===
def load_glioma_dataset(path, size=64):
    X = []
    for fname in os.listdir(path):
        img = cv2.imread(os.path.join(path, fname), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (size, size)).flatten()
            X.append(img)
    X = np.array(X)
    y = np.zeros(X.shape[0])
    y[X.shape[0] // 2:] = 1  # Artificial 2-class
    return X, y

# === A1: Correlation Heatmap ===
def plot_correlation_heatmap(X):
    X_df = pd.DataFrame(X[:, :100])  # Limit to 100 features to keep plot readable
    corr = X_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap='coolwarm')
    plt.title("A1: Feature Correlation Heatmap")
    plt.show()

# === A2/A3: PCA-based Classification ===
def run_pca_classification(X, y, variance_threshold=0.99):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=variance_threshold)
    X_pca = pca.fit_transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42, stratify=y)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n--- PCA {int(variance_threshold*100)}% Variance Retained ---")
    print(f"Number of PCA features: {X_pca.shape[1]}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return model, X_train, X_test, y_train, y_test, pca

# === A4: Sequential Feature Selector ===
def run_sequential_selection(X, y, n_features=50):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(random_state=42)
    sfs = SequentialFeatureSelector(model, n_features_to_select=n_features, direction='forward', cv=3, n_jobs=-1)
    sfs.fit(X_scaled, y)
    
    X_reduced = sfs.transform(X_scaled)
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42, stratify=y)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n--- A4: Sequential Feature Selection (Top {n_features} Features) ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return model, X_train, X_test, y_train, y_test

# === A5: LIME + SHAP Explanation ===
def explain_with_lime_shap(model, X_train, X_test):
    print("\n--- A5: LIME & SHAP Explainability ---")

    # LIME
    lime_exp = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        mode='classification',
        feature_names=[f'pixel_{i}' for i in range(X_train.shape[1])],
        class_names=['Class 0', 'Class 1']
    )
    lime_explanation = lime_exp.explain_instance(X_test[0], model.predict_proba)
    lime_explanation.save_to_file("lime_explanation.html")
    print("LIME explanation saved to lime_explanation.html")

    # SHAP
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test[:10])
    shap.summary_plot(shap_values, X_test[:10], feature_names=[f'pixel_{i}' for i in range(X_train.shape[1])])

# === MAIN EXECUTION ===
if __name__ == "__main__":
    import pandas as pd
    X, y = load_glioma_dataset(DATASET_PATH)

    # A1: Correlation Heatmap
    plot_correlation_heatmap(X)

    # A2: PCA with 99% variance
    model_99, X_train_99, X_test_99, y_train_99, y_test_99, _ = run_pca_classification(X, y, 0.99)

    # A3: PCA with 95% variance
    model_95, X_train_95, X_test_95, y_train_95, y_test_95, _ = run_pca_classification(X, y, 0.95)

    # A4: Sequential Feature Selection
    model_sfs, X_train_sfs, X_test_sfs, y_train_sfs, y_test_sfs = run_sequential_selection(X, y)

    # A5: Explainability
    explain_with_lime_shap(model_sfs, X_train_sfs, X_test_sfs)
