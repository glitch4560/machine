import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
import lime
import lime.lime_tabular

# === Path to dataset ===
DATASET_PATH = "C:\\AIO\\Semster Files\\SEMSTER - 4\\ML\\Lab Work\\ML_Assignment_09_BL.EN.U4AIE23138\\Dataset"

# === Function to load and split into 2 classes (artificial for small dataset) ===
def load_dataset(path: str, size: int = 64):
    images = []
    for fname in os.listdir(path):
        img_path = os.path.join(path, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (size, size)).flatten()
            images.append(img)
    
    features = np.array(images)
    count = len(features)

    # Artificially assign half as class 0 and half as class 1
    labels = np.zeros(count, dtype=int)
    labels[count // 2:] = 1
    return features, labels

# === A1: Stacking Classifier ===
def create_stacking_classifier(X_train, y_train):
    base_classifiers = [
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('svm', SVC(probability=True, random_state=42)),
        ('mlp', MLPClassifier(max_iter=300, random_state=42)),
        ('gnb', GaussianNB())
    ]
    final_estimator = LogisticRegression()

    model = StackingClassifier(
        estimators=base_classifiers,
        final_estimator=final_estimator,
        cv=2  # Fix: Use 2-fold CV due to small dataset
    )
    model.fit(X_train, y_train)
    return model

# === A2: Pipeline ===
def create_pipeline(X_train, y_train):
    pipe = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    return pipe

# === A3: LIME ===
def explain_with_lime(pipeline, X_train, X_test):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=[f'pixel_{i}' for i in range(X_train.shape[1])],
        class_names=['Glioma_0', 'Glioma_1'],
        mode='classification'
    )
    explanation = explainer.explain_instance(X_test[0], pipeline.predict_proba)

    # Save explanation as HTML with proper encoding
    html_path = "lime_explanation.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(explanation.as_html())
    
    print(f"LIME explanation saved to: {html_path}")
    print("Open this file in your browser to view the explanation.")
    return html_path


# === MAIN ===
if __name__ == "__main__":
    X, y = load_dataset(DATASET_PATH)
    
    if len(np.unique(y)) < 2:
        raise ValueError("Dataset must have at least two classes to perform classification.")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # A1
    print("\n--- A1: Stacking Classifier ---")
    stacking_model = create_stacking_classifier(X_train, y_train)
    y_pred_stack = stacking_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred_stack))
    print(classification_report(y_test, y_pred_stack))

    # A2
    print("\n--- A2: Pipeline with MLP ---")
    pipeline_model = create_pipeline(X_train, y_train)
    y_pred_pipe = pipeline_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred_pipe))
    print(classification_report(y_test, y_pred_pipe))

    # A3
    print("\n--- A3: LIME Explanation ---")
    explain_with_lime(pipeline_model, X_train, X_test)
