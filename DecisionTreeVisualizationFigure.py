import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

# Load dataset
df = pd.read_csv(path+"Enc_dataset.csv")  # Change the filename if necessary
# df = pd.read_csv(path+"BSW07_experiment_results.csv")

# Select rows where "Attributes" are in the given list
#selected_attributes = [2, 4, 5, 9, 12, 20, 30, 50, 100, 250, 500]
#df_selected = df[df["Attributes"].isin(selected_attributes)]

# Remove rows where Attributes = 500
df_filtered = df[df["Attributes"] != 500]
df =  df_filtered
#df = df_selected

# Define numerical and categorical features
numerical_features = [
    "File_Size_Bytes", "Attributes", "Avail_CPU_%", "Avail_Memory_%",
    "Full_Execution_Time_s", "Partial_RP1_Time_s", "Transmission_Time_s",
    "Partial_RP2_Time_s", "Partial_Total_Time_s"
]
categorical_features = ["CPU_State", "Memory_State", "Network"]
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

# Transform features using the preprocessor
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Train Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_classifier.fit(X_train_transformed, y_train)

# Visualizing the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(
    dt_classifier,
    filled=True,
    feature_names=preprocessor.get_feature_names_out(),
    class_names=["No Offload", "Offload"]
)
plt.title("Decision Tree Visualization")
plt.show()
