import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("Financial Transactions.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# =====================================================
# MISSING VALUES
# =====================================================

print("\nMissing Values")
print(df.isnull().sum())

df.dropna(inplace=True)

# =====================================================
# ENCODE TYPE COLUMN
# =====================================================

encoder = LabelEncoder()

df["Type"] = encoder.fit_transform(df["Type"])

# =====================================================
# DROP HIGH CARDINALITY COLUMNS
# =====================================================

drop_cols = ["NameOrig", "NameDest"]

for col in drop_cols:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# =====================================================
# HANDLE DATETIME
# =====================================================

if "TransactionTime" in df.columns:

    df["TransactionTime"] = pd.to_datetime(
        df["TransactionTime"],
        errors="coerce"
    )

    df["Hour"] = df["TransactionTime"].dt.hour
    df["Day"] = df["TransactionTime"].dt.day
    df["Month"] = df["TransactionTime"].dt.month

    df.drop("TransactionTime", axis=1, inplace=True)

# =====================================================
# FRAUD DISTRIBUTION
# =====================================================

plt.figure(figsize=(8,5))

sns.countplot(x="IsFraud", data=df)

plt.title("Fraud Distribution")

plt.savefig("fraud_distribution.png")

plt.show()

# =====================================================
# AMOUNT DISTRIBUTION
# =====================================================

plt.figure(figsize=(10,5))

sns.histplot(df["Amount"], bins=30, kde=True)

plt.title("Transaction Amount Distribution")

plt.savefig("amount_distribution.png")

plt.show()

# =====================================================
# CORRELATION HEATMAP
# =====================================================

plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(numeric_only=True),
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig("correlation_heatmap.png")

plt.show()

# =====================================================
# FEATURES AND TARGET
# =====================================================

X = df.drop("IsFraud", axis=1)

y = df["IsFraud"]

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# =====================================================
# ISOLATION FOREST
# =====================================================

print("\nIsolation Forest")

iso_model = IsolationForest(
    contamination=0.02,
    random_state=42
)

iso_model.fit(X_train)

iso_pred = iso_model.predict(X_test)

iso_pred = np.where(
    iso_pred == -1,
    1,
    0
)

iso_precision = precision_score(y_test, iso_pred)
iso_recall = recall_score(y_test, iso_pred)
iso_f1 = f1_score(y_test, iso_pred)

print("Precision:", iso_precision)
print("Recall:", iso_recall)
print("F1:", iso_f1)

# =====================================================
# ONE CLASS SVM (SAMPLED)
# =====================================================

print("\nOne Class SVM")

normal_data = X_train[y_train == 0]

sample_size = min(
    5000,
    len(normal_data)
)

normal_sample = normal_data.sample(
    sample_size,
    random_state=42
)

svm_model = OneClassSVM(
    kernel="rbf",
    gamma="auto",
    nu=0.02
)

svm_model.fit(normal_sample)

svm_pred = svm_model.predict(X_test)

svm_pred = np.where(
    svm_pred == -1,
    1,
    0
)

svm_precision = precision_score(y_test, svm_pred)
svm_recall = recall_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred)

print("Precision:", svm_precision)
print("Recall:", svm_recall)
print("F1:", svm_f1)

# =====================================================
# RANDOM FOREST
# =====================================================

print("\nRandom Forest")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

rf_precision = precision_score(
    y_test,
    rf_pred
)

rf_recall = recall_score(
    y_test,
    rf_pred
)

rf_f1 = f1_score(
    y_test,
    rf_pred
)

print("Accuracy :", rf_accuracy)
print("Precision:", rf_precision)
print("Recall   :", rf_recall)
print("F1 Score :", rf_f1)

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

print("\nClassification Report")

print(
    classification_report(
        y_test,
        rf_pred
    )
)

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_test,
    rf_pred
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.title("Random Forest Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")

plt.show()

# =====================================================
# MODEL PERFORMANCE TABLE
# =====================================================

performance_df = pd.DataFrame({

    "Model":[
        "Isolation Forest",
        "One Class SVM",
        "Random Forest"
    ],

    "Precision":[
        iso_precision,
        svm_precision,
        rf_precision
    ],

    "Recall":[
        iso_recall,
        svm_recall,
        rf_recall
    ],

    "F1_Score":[
        iso_f1,
        svm_f1,
        rf_f1
    ]
})

print("\nModel Comparison")

print(performance_df)

performance_df.to_csv(
    "model_performance_dashboard.csv",
    index=False
)

# =====================================================
# POWER BI DATASET
# =====================================================

powerbi_df = X_test.copy()

powerbi_df["Actual_Fraud"] = y_test.values

powerbi_df["IsolationForest"] = iso_pred

powerbi_df["OneClassSVM"] = svm_pred

powerbi_df["RandomForest"] = rf_pred

powerbi_df.to_csv(
    "powerbi_fraud_dashboard.csv",
    index=False
)

print("\nFILES GENERATED")

print("fraud_distribution.png")
print("amount_distribution.png")
print("correlation_heatmap.png")
print("confusion_matrix.png")

print("powerbi_fraud_dashboard.csv")
print("model_performance_dashboard.csv")

print("\nPROJECT COMPLETED")