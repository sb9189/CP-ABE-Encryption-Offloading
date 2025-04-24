import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from google.colab import drive
drive.mount('/content/drive')

# uploaded = files.upload()

# Read the dataset
# file_name = "Enc_dataset.csv"  # Change this if the filename is different
# df = pd.read_csv(file_name)

path = "/content/drive/MyDrive/Colab Notebooks/sb/"  # Change this if the filename is different
df = pd.read_csv(path+"Enc_dataset.csv")  # Change the filename if necessary
#df = pd.read_csv(path+"BSW07_experiment_results.csv")

# print('Before: ', df.info())
# Select rows where "Attributes" are in the given list
selected_attributes = [2, 4, 5, 9, 12, 20, 30, 50, 100, 250, 500]
df_selected = df[df["Attributes"].isin(selected_attributes)]
df  = df_selected
# print('After: ', df.info())
# Identify numerical and categorical columns
numerical_features = [
    "File_Size_Bytes", "Attributes", "Avail_CPU_%", "Avail_Memory_%",
    "Full_Execution_Time_s", "Partial_RP1_Time_s", "Transmission_Time_s",
    "Partial_RP2_Time_s", "Partial_Total_Time_s"
]
categorical_features = ["CPU_State", "Memory_State", "Network"]

# Define target variable
target = "Offload_Decision"

# Preprocessing: StandardScaler for numerical, OneHotEncoder for categorical
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Split the dataset
X = df.drop(columns=[target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### 1. Support Vector Machine (SVM) with RBF Kernel
svm_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC(kernel="rbf", C=1, gamma="scale"))
])
#svm_model.fit(X_train, y_train)
#svm_pred = svm_model.predict(X_test)

### 2. k-Nearest Neighbors (k-NN) with Hyperparameter Tuning
knn_params = {'classifier__n_neighbors': [3, 5, 7, 9, 11]}
knn_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", KNeighborsClassifier())
])
knn_grid = GridSearchCV(knn_model, knn_params, cv=5)
knn_grid.fit(X_train, y_train)
knn_pred = knn_grid.best_estimator_.predict(X_test)

### 3. Decision Tree with Depth Restriction to Avoid Overfitting
dt_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(max_depth=10, random_state=42))
])
#dt_model.fit(X_train, y_train)
#dt_pred = dt_model.predict(X_test)

### 4. Random Forest as an Ensemble Alternative for Better Generalization
rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
])
#rf_model.fit(X_train, y_train)
#rf_pred = rf_model.predict(X_test)
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
