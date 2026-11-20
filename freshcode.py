
# ==========================================================
# CUSTOMER CHURN PREDICTION
# DAY 44 TO DAY 52 - COMPLETE ML PROJECT
# ==========================================================


# ----------------------------------------------------------
# 1. IMPORT LIBRARIES
# ----------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)


# ----------------------------------------------------------
# 2. DATASET LOADING
# ----------------------------------------------------------

data = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

print("Dataset Shape:", data.shape)

print("\nFirst Five Rows:")
print(data.head())


# ----------------------------------------------------------
# 3. DATASET UNDERSTANDING
# ----------------------------------------------------------

print("\nDataset Information:")
data.info()

print("\nChurn Distribution:")
print(data["Churn"].value_counts())

print("\nChurn Percentage:")
print(data["Churn"].value_counts(normalize=True) * 100)

print("\nStatistical Summary:")
print(data.describe())


# ----------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS
# ----------------------------------------------------------

print("\nGender vs Churn:")
print(data.groupby("gender")["Churn"].value_counts())

print("\nContract vs Churn:")
print(data.groupby("Contract")["Churn"].value_counts())

print("\nPayment Method vs Churn:")
print(data.groupby("PaymentMethod")["Churn"].value_counts())

print("\nAverage Monthly Charges:")
print(data.groupby("Churn")["MonthlyCharges"].mean())


# Churn Distribution Graph

churn_count = data["Churn"].value_counts()

plt.bar(churn_count.index, churn_count.values)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn Status")
plt.ylabel("Number of Customers")

plt.show()


# Monthly Charges Histogram

plt.hist(data["MonthlyCharges"], bins=20)

plt.title("Monthly Charges Distribution")
plt.xlabel("Monthly Charges")
plt.ylabel("Number of Customers")

plt.show()


# ----------------------------------------------------------
# 5. DATA CLEANING
# ----------------------------------------------------------

# TotalCharges mein blank values check karna

blank_values = data["TotalCharges"].str.strip() == ""

print("\nBlank TotalCharges:", blank_values.sum())


# Blank values ko 0 se replace karna

data.loc[blank_values, "TotalCharges"] = "0"


# TotalCharges ko numerical format mein convert karna

data["TotalCharges"] = pd.to_numeric(data["TotalCharges"])


# CustomerID remove karna

data = data.drop("customerID", axis=1)


# Missing values check

print("\nMissing Values:")
print(data.isnull().sum().sum())


# Duplicate rows check

print("\nDuplicate Rows:", data.duplicated().sum())


# Cleaned dataset save karna

data.to_csv(
    "data/cleaned_customer_churn.csv",
    index=False
)

print("\nCleaned Dataset Saved Successfully!")


# ----------------------------------------------------------
# 6. FEATURES (X) AND TARGET (y)
# ----------------------------------------------------------

X = data.drop("Churn", axis=1)

y = data["Churn"]


print("\nX Shape:", X.shape)
print("y Shape:", y.shape)


# ----------------------------------------------------------
# 7. TRAIN TEST SPLIT
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


# ----------------------------------------------------------
# 8. NUMERICAL AND CATEGORICAL COLUMNS
# ----------------------------------------------------------

numeric_columns = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_columns = X.select_dtypes(
    include="object"
).columns.tolist()


# ----------------------------------------------------------
# 9. PREPROCESSING
# ----------------------------------------------------------

# Numerical data ke liye StandardScaler

numeric_transformer = StandardScaler()


# Categorical data ke liye OneHotEncoder

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)


# ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numeric_transformer, numeric_columns),
        ("categorical", categorical_transformer, categorical_columns)
    ]
)


# ----------------------------------------------------------
# 10. LOGISTIC REGRESSION MODEL
# ----------------------------------------------------------

log_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ]
)


log_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test)


log_accuracy = accuracy_score(y_test, log_pred)

print("\nLogistic Regression Accuracy:")
print(log_accuracy * 100, "%")


# ----------------------------------------------------------
# 11. DECISION TREE MODEL
# ----------------------------------------------------------

dt_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier(random_state=42))
    ]
)


dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)


dt_accuracy = accuracy_score(y_test, dt_pred)

print("\nDecision Tree Accuracy:")
print(dt_accuracy * 100, "%")


# ----------------------------------------------------------
# 12. RANDOM FOREST MODEL
# ----------------------------------------------------------

rf_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(random_state=42))
    ]
)


rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)


rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:")
print(rf_accuracy * 100, "%")


# ----------------------------------------------------------
# 13. MODEL COMPARISON
# ----------------------------------------------------------

print("\n========== MODEL COMPARISON ==========")

print("Logistic Regression:", log_accuracy * 100, "%")

print("Decision Tree:", dt_accuracy * 100, "%")

print("Random Forest:", rf_accuracy * 100, "%")


# ----------------------------------------------------------
# 14. LOGISTIC REGRESSION EVALUATION
# ----------------------------------------------------------

print("\n========== LOGISTIC REGRESSION EVALUATION ==========")


print("\nConfusion Matrix:")

log_cm = confusion_matrix(
    y_test,
    log_pred,
    labels=["No", "Yes"]
)

print(log_cm)


log_precision = precision_score(
    y_test,
    log_pred,
    pos_label="Yes"
)

log_recall = recall_score(
    y_test,
    log_pred,
    pos_label="Yes"
)

log_f1 = f1_score(
    y_test,
    log_pred,
    pos_label="Yes"
)


print("\nAccuracy:", log_accuracy * 100, "%")

print("Precision:", log_precision * 100, "%")

print("Recall:", log_recall * 100, "%")

print("F1 Score:", log_f1 * 100, "%")


print("\nClassification Report:")

print(classification_report(y_test, log_pred))


# ----------------------------------------------------------
# 15. ROC CURVE AND ROC-AUC
# ----------------------------------------------------------

# Yes class ki probability nikalna

log_probability = log_model.predict_proba(X_test)[:, 1]


# ROC Curve

fpr, tpr, thresholds = roc_curve(
    y_test,
    log_probability,
    pos_label="Yes"
)


plt.plot(
    fpr,
    tpr,
    label="Logistic Regression"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Guess"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()


# ROC-AUC

log_auc = roc_auc_score(
    y_test,
    log_probability
)

print("\nROC-AUC:", log_auc)


# ----------------------------------------------------------
# 16. OVERFITTING CHECK
# ----------------------------------------------------------

print("\n========== OVERFITTING CHECK ==========")


# Logistic Regression

log_train_pred = log_model.predict(X_train)

log_test_pred = log_model.predict(X_test)


log_train_accuracy = accuracy_score(
    y_train,
    log_train_pred
)

log_test_accuracy = accuracy_score(
    y_test,
    log_test_pred
)


print("\nLogistic Regression")

print("Training Accuracy:", log_train_accuracy * 100, "%")

print("Testing Accuracy:", log_test_accuracy * 100, "%")


# Decision Tree

dt_train_pred = dt_model.predict(X_train)

dt_test_pred = dt_model.predict(X_test)


dt_train_accuracy = accuracy_score(
    y_train,
    dt_train_pred
)

dt_test_accuracy = accuracy_score(
    y_test,
    dt_test_pred
)


print("\nDecision Tree")

print("Training Accuracy:", dt_train_accuracy * 100, "%")

print("Testing Accuracy:", dt_test_accuracy * 100, "%")


# Random Forest

rf_train_pred = rf_model.predict(X_train)

rf_test_pred = rf_model.predict(X_test)


rf_train_accuracy = accuracy_score(
    y_train,
    rf_train_pred
)

rf_test_accuracy = accuracy_score(
    y_test,
    rf_test_pred
)


print("\nRandom Forest")

print("Training Accuracy:", rf_train_accuracy * 100, "%")

print("Testing Accuracy:", rf_test_accuracy * 100, "%")


# ----------------------------------------------------------
# 17. CROSS VALIDATION
# ----------------------------------------------------------

print("\n========== CROSS VALIDATION ==========")


cv_scores = cross_val_score(
    log_model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)


print("Cross Validation Scores:", cv_scores)

print(
    "Average CV Accuracy:",
    cv_scores.mean() * 100,
    "%"
)


# ----------------------------------------------------------
# 18. GRID SEARCH CV
# ----------------------------------------------------------

print("\n========== GRID SEARCH CV ==========")


dt_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier(random_state=42))
    ]
)


param_grid = {
    "model__max_depth": [3, 5, 7],
    "model__min_samples_split": [2, 5]
}


grid_search = GridSearchCV(
    estimator=dt_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy"
)


grid_search.fit(X_train, y_train)


print("\nBest Grid Parameters:")
print(grid_search.best_params_)


print("\nBest Grid CV Accuracy:")
print(grid_search.best_score_ * 100, "%")


best_dt_model = grid_search.best_estimator_


best_dt_pred = best_dt_model.predict(X_test)


best_dt_accuracy = accuracy_score(
    y_test,
    best_dt_pred
)


print("\nTuned Decision Tree Test Accuracy:")
print(best_dt_accuracy * 100, "%")


# ----------------------------------------------------------
# 19. RANDOMIZED SEARCH CV
# ----------------------------------------------------------

print("\n========== RANDOMIZED SEARCH CV ==========")


param_distributions = {
    "model__max_depth": [3, 5, 7, 10],
    "model__min_samples_split": [2, 5, 10, 15]
}


random_search = RandomizedSearchCV(
    estimator=dt_pipeline,
    param_distributions=param_distributions,
    n_iter=5,
    cv=5,
    scoring="accuracy",
    random_state=42
)


random_search.fit(X_train, y_train)


print("\nBest Randomized Parameters:")
print(random_search.best_params_)


print("\nBest Randomized CV Accuracy:")
print(random_search.best_score_ * 100, "%")


best_random_model = random_search.best_estimator_


best_random_pred = best_random_model.predict(X_test)


random_test_accuracy = accuracy_score(
    y_test,
    best_random_pred
)


print("\nRandomizedSearchCV Test Accuracy:")
print(random_test_accuracy * 100, "%")


# ----------------------------------------------------------
# 20. FINAL MODEL TRAINING
# ----------------------------------------------------------

print("\n========== FINAL MODEL ==========")


# Logistic Regression ko final candidate select karna

final_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ]
)


# Final model train karna

final_model.fit(X_train, y_train)


# Predictions generate karna

final_predictions = final_model.predict(X_test)


# Final accuracy

final_model_accuracy = accuracy_score(
    y_test,
    final_predictions
)


print("\nFinal Model Accuracy:")

print(final_model_accuracy * 100, "%")


# ----------------------------------------------------------
# 21. SAVE FINAL MODEL
# ----------------------------------------------------------

# Models folder create karna

os.makedirs("models", exist_ok=True)


# Final model save karna

joblib.dump(
    final_model,
    "models/customer_churn_model.pkl"
)


print("\nFinal Model Saved Successfully!")


# ----------------------------------------------------------
# 22. LOAD SAVED MODEL
# ----------------------------------------------------------

loaded_model = joblib.load(
    "models/customer_churn_model.pkl"
)


# Loaded model se predictions

loaded_predictions = loaded_model.predict(X_test)


print("\nLoaded Model Predictions:")

print(loaded_predictions[:10])


print("\nCustomer Churn Prediction Project Completed!")
