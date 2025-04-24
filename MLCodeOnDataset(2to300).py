import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from google.colab import files
import time
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
path = '/content/drive/MyDrive/Colab Notebooks/sb/'
# uploaded = files.upload()

df = pd.read_csv(path+"Enc_dataset.csv")  # Change the filename if necessary

# Remove rows where Attributes = 500
df_filtered = df[df["Attributes"] != 500]
df =  df_filtered
# Define target variable
target = "Offload_Decision"

# Identify numerical and categorical columns
numerical_features = [
    "File_Size_Bytes", "Attributes", "Avail_CPU_%", "Avail_Memory_%",
    "Full_Execution_Time_s", "Partial_RP1_Time_s", "Transmission_Time_s",
    "Partial_RP2_Time_s", "Partial_Total_Time_s"
]
categorical_features = ["CPU_State", "Memory_State", "Network"]

# Split dataset
X = df[numerical_features + categorical_features]
y = df[target]

# Apply OneHotEncoding to categorical data
encoder = OneHotEncoder(handle_unknown="ignore")
X_encoded = encoder.fit_transform(X[categorical_features])

# Grid Search for Best Correlation Threshold
best_accuracy = 0
best_threshold = None
best_selected_features = None

for threshold in np.arange(0.75, 0.96, 0.05):  # Testing correlation thresholds from 0.75 to 0.95
    # Compute correlation matrix
    corr_matrix = df[numerical_features + [target]].corr()

    # Identify highly correlated features
    high_corr_features = set()
    corr_pairs = corr_matrix.abs().unstack().sort_values(ascending=False)

    for (feature1, feature2), correlation in corr_pairs.items():
        if feature1 != feature2 and correlation > threshold:
            high_corr_features.add(feature2)  # Remove the second feature to avoid redundancy

    # Select non-redundant features
    selected_features = [f for f in numerical_features if f not in high_corr_features]

    # Prepare dataset with selected features
    X_selected = df[selected_features + categorical_features]

    # **Fix: Re-fit StandardScaler on reduced numerical features**
    scaler = StandardScaler()
    X_selected_scaled = scaler.fit_transform(X_selected[selected_features])  # Now fits only selected features

    # Encode categorical features
    X_selected_encoded = encoder.transform(X_selected[categorical_features])

    # Combine scaled and encoded features
    X_selected_final = np.hstack((X_selected_scaled, X_selected_encoded.toarray()))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_selected_final, y, test_size=0.2, random_state=42)

    # Train a Random Forest Model (since it generalizes well)
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=3, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, rf_pred)

    # Track best threshold
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold
        best_selected_features = selected_features

# # Final Training with Best Features
# print(f"Best Correlation Threshold: {best_threshold}")
# print(f"Selected Features After Correlation Filtering: {best_selected_features}")

# Prepare dataset with the best correlation threshold
X_final_selected = df[best_selected_features + categorical_features]

# **Fix: Re-fit StandardScaler again on final selected numerical features**
scaler = StandardScaler()
X_scaled_best = scaler.fit_transform(X_final_selected[best_selected_features])

# Encode categorical features
X_encoded_best = encoder.transform(X_final_selected[categorical_features])

# Combine scaled and encoded features
X_final_best = np.hstack((X_scaled_best, X_encoded_best.toarray()))

# Train-test split with best features
X_train, X_test, y_train, y_test = train_test_split(X_final_best, y, test_size=0.2, random_state=42)

# Train models
svm_model = SVC(kernel="linear")
knn_model = KNeighborsClassifier(n_neighbors=5)
dt_model = DecisionTreeClassifier(max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=3, random_state=42)

svm_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# # Predictions
# svm_pred = svm_model.predict(X_test)
# knn_pred = knn_model.predict(X_test)
# dt_pred = dt_model.predict(X_test)
# rf_pred = rf_model.predict(X_test)


# --- Train & Time ---
start = time.time()
svm_model.fit(X_train, y_train)
svm_train_time = time.time() - start

start = time.time()
knn_model.fit(X_train, y_train)
knn_train_time = time.time() - start

start = time.time()
dt_model.fit(X_train, y_train)
dt_train_time = time.time() - start

start = time.time()
rf_model.fit(X_train, y_train)
rf_train_time = time.time() - start

# --- Inference & Time ---
start = time.time()
svm_pred = svm_model.predict(X_test)
svm_infer_time = time.time() - start

start = time.time()
knn_pred = knn_model.predict(X_test)
knn_infer_time = time.time() - start

start = time.time()
dt_pred = dt_model.predict(X_test)
dt_infer_time = time.time() - start

start = time.time()
rf_pred = rf_model.predict(X_test)
rf_infer_time = time.time() - start

# --- Output Results ---
print(f"SVM     -> Train: {svm_train_time:.6f} sec | Inference: {svm_infer_time:.6f} sec")
print(f"KNN     -> Train: {knn_train_time:.6f} sec | Inference: {knn_infer_time:.6f} sec")
print(f"DT      -> Train: {dt_train_time:.6f} sec | Inference: {dt_infer_time:.6f} sec")
print(f"RF      -> Train: {rf_train_time:.6f} sec | Inference: {rf_infer_time:.6f} sec")

# Accuracy scores
accuracy_results = {
    "SVM (RBF) Accuracy": accuracy_score(y_test, svm_pred),
    "SVM (RBF) Precision": precision_score(y_test, svm_pred, average='weighted', zero_division=0),
    "SVM (RBF) Recall": recall_score(y_test, svm_pred, average='weighted', zero_division=0),
    "SVM (RBF) F1-Score": f1_score(y_test, svm_pred, average='weighted', zero_division=0),
    "k-NN Accuracy": accuracy_score(y_test, knn_pred),
    "k-NN Precision": precision_score(y_test, knn_pred, average='weighted', zero_division=0),
    "k-NN Recall": recall_score(y_test, knn_pred, average='weighted', zero_division=0),
    "k-NN F1-Score": f1_score(y_test, knn_pred, average='weighted', zero_division=0),
    "Decision Tree (Depth=10) Accuracy": accuracy_score(y_test, dt_pred),
    "Decision Tree Precision": precision_score(y_test, dt_pred, average='weighted', zero_division=0),
    "Decision Tree Recall": recall_score(y_test, dt_pred, average='weighted', zero_division=0),
    "Decision Tree F1-Score": f1_score(y_test, dt_pred, average='weighted', zero_division=0),
    "Random Forest Accuracy": accuracy_score(y_test, rf_pred),
    "Random Forest Precision": precision_score(y_test, rf_pred, average='weighted', zero_division=0),
    "Random Forest Recall": recall_score(y_test, rf_pred, average='weighted', zero_division=0),
    "Random Forest F1-Score": f1_score(y_test, rf_pred, average='weighted', zero_division=0)
}

# Print final accuracy results
print("\nFinal Model Performance After Grid Search on Correlation Threshold:\n", accuracy_results)
