import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Dataset load kar rahe hain
data = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Dataset ki first 5 rows display kar rahe hain
# print(data.head())


# Dataset mein total rows aur columns check karna
# print("Dataset Shape:", data.shape)

# # Total rows check karna
# print("Total Customers:", data.shape[0])

# # Total columns check karna
# print("Total Columns:", data.shape[1])

# print("DATA_Column_name:")
# print(data.columns)


# Churn column mein Yes aur No ki counting
# print(data["Churn"].value_counts())


# # Churn ka percentage distribution
# print(data["Churn"].value_counts(normalize=True) * 100)

# Dataset ke sabhi columns ke data types check karna
# print(data.dtypes)

# pure data ki info dena
# data.info()

# # Har column mein missing values count karna
# print(data.isnull().sum())

# # Complete dataset mein total missing values count karna
# print("Total Missing Values:", data.isnull().sum().sum())

# Har column mein empty strings count karna
# print("Blank values in TotalCharges:")
# print((data["TotalCharges"].str.strip() == "").sum())

# Numerical columns ki statistical summary display karna
# print(data.describe())


# # Male aur Female customers ko alag groups mein divide karna
# gender_groups = data.groupby("gender")

# # Har gender mein Churn Yes aur No count karna
# print(gender_groups["Churn"].value_counts())

# contract_group = data.groupby("Contract")

# print(contract_group["Churn"].value_counts())

# tenure_groups = data.groupby("tenure")

# print(tenure_groups["Churn"].value_counts())

# # Tenure 0 months wale customers
# tenure_zero = data[data["tenure"] == 0]

# # Total customers count
# print("Total customers:", len(tenure_zero))

# print(tenure_zero["Churn"].value_counts())

# tenure_one = data[data["tenure"] == 1]

# print("Total customers (1):", len(tenure_one))

# print(tenure_one["Churn"].value_counts())


# tenure_two = data[data["tenure"] == 2]
# print("Total customer(2): ", len(tenure_two))
# print(tenure_two["Churn"].value_counts())


# monthly charge group

# monthly_groups = data.groupby("MonthlyCharges")

# print(monthly_groups["Churn"].value_counts())

# Churn ke according average monthly charges
# print(data.groupby("Churn")["MonthlyCharges"].mean())


# PaymentMethod vs Churn.

# payment_groups = data.groupby("PaymentMethod")
# print(payment_groups["Churn"].value_counts())

churn_count = data["Churn"].value_counts()


# bar chart

# plt.bar(churn_count.index, churn_count.values)

# plt.title("Customer Churn Distribution")
# plt.xlabel("Churn Status")
# plt.ylabel("Number of customers")


# plt.show()


# histogram

# plt.hist(data["MonthlyCharges"], bins=20)

# plt.title("Monthly Charges Distribution")
# plt.xlabel("Monthly Charges")
# plt.ylabel("Number of Customers")

# plt.show()


# Monthly charges ka box plot
# plt.boxplot(data["MonthlyCharges"])

# plt.title("Monthly Charges Box Plot")
# plt.ylabel("Monthly Charges")

# plt.show()


# missing value

# print(data.isnull().sum())

# TotalCharges mein blank values check karna
blank_values = data["TotalCharges"].str.strip() == ""

# print("Blank values:", blank_values.sum())


# Blank TotalCharges wale customers ka complete record
# print(data[blank_values])

data.loc[blank_values, "TotalCharges"] = "0"

blank_values = data["TotalCharges"].str.strip() == ""

# print("Remaining blank values:", blank_values.sum())


# print(data["TotalCharges"].dtype)


# TotalCharges ko numerical format mein convert karna
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"])

# print(data["TotalCharges"].dtype)


# print("Duplicated Row: ",data.duplicated().sum())

# customerID column remove karna
data = data.drop("customerID", axis=1)

# Remaining columns check karna
# print(data.columns)
# print(data.shape)

data.to_csv("data/cleaned_customer_churn.csv", index=False)

# print("Cleaned dataset saved successfully!")


# ## Final dataset information
# print(data.info())

# # Missing values check
# print(data.isnull().sum())

# # Duplicate rows check
# print("Duplicate rows:", data.duplicated().sum())

# # Dataset shape
# print("Final shape:", data.shape)


# now We Start making X data and y Data

X = data.drop("Churn", axis=1)

y = data["Churn"]

# print("X shape: ",X.shape)
# print("y shape: ",y.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# print("X_train:", X_train.shape)
# print("X_test:", X_test.shape)
# print("y_train:", y_train.shape)
# print("y_test:", y_test.shape)

# print("Training Churn Distribution:")
# print(y_train.value_counts(normalize=True) * 100)

# print("Testing Churn Distribution:")
# print(y_test.value_counts(normalize=True) * 100)


scaler = StandardScaler()

numeric_columns = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

# print(X_train[numeric_columns].head())

X_train_scaled = X_train.copy()

X_train_scaled[numeric_columns] = scaler.fit_transform(
    X_train[numeric_columns])

# print(X_train_scaled[numeric_columns].head())

X_test_scaled = X_test.copy()

X_test_scaled[numeric_columns] = scaler.transform(X_test[numeric_columns])

# print(X_test_scaled[numeric_columns].head())


# print("Training Data Mean:")
# print(X_train_scaled[numeric_columns].mean())

# print("\nTraining Data Standard Deviation:")
# print(X_train_scaled[numeric_columns].std())

# print("Training Data Mean:")
# print(X_train_scaled[numeric_columns].mean())


categorical_columns = X.select_dtypes(include="object").columns

# print("Categorical Columns:")
# print(categorical_columns)


# One-Hot Encoding

# Encoder create karna
encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

# Training data ke categorical columns select karna
X_train_categorical = X_train[categorical_columns]

# Categories learn karke 0 aur 1 mein convert karna
X_train_encoded = encoder.fit_transform(X_train_categorical)

# Output check karna
# print("Encoded Training Shape:", X_train_encoded.shape)

# Testing data ke categorical columns select karna
X_test_categorical = X_test[categorical_columns]

# Training data se learned categories use karke testing data encode karna
X_test_encoded = encoder.transform(X_test_categorical)

# Output check karna
# print("Encoded Testing Shape:", X_test_encoded.shape)

X_train_numerical = X_train_scaled[
    numeric_columns
]

# print("Numerical Shape:", X_train_numerical.shape)

X_train_numerical = X_train_scaled.drop(
    categorical_columns,
    axis=1
)

# print("Numerical Shape:", X_train_numerical.shape)
# print(X_train_numerical.head())


encoded_columns = encoder.get_feature_names_out(categorical_columns)

# print(encoded_columns)


X_train_encoded_df = pd.DataFrame(
    X_train_encoded,
    columns=encoded_columns,
    index=X_train.index
)

# print("Encoded DataFrame Shape:", X_train_encoded_df.shape)
# print(X_train_encoded_df.head())

X_train_final = pd.concat(
    [X_train_numerical, X_train_encoded_df],
    axis=1
)

# print("Final Training Shape:", X_train_final.shape)


X_test_numerical = X_test_scaled.drop(
    categorical_columns,
    axis=1
)

# print("Testing Numerical Shape:", X_test_numerical.shape)

X_test_encoded_df = pd.DataFrame(
    X_test_encoded,
    columns=encoded_columns,
    index=X_test.index
)

# print("Testing Encoded Shape:", X_test_encoded_df.shape)

X_test_final = pd.concat(
    [X_test_numerical, X_test_encoded_df],
    axis=1
)

# print("Final Testing Shape:", X_test_final.shape)

# print("Training Missing Values:")
# print(X_train_final.isnull().sum().sum())

# print("\nTesting Missing Values:")
# print(X_test_final.isnull().sum().sum())

# print("Training Data Types:")
# print(X_train_final.dtypes.value_counts())

# print("\nTesting Data Types:")
# print(X_test_final.dtypes.value_counts())


model = LogisticRegression()

model.fit(X_train_final, y_train)

y_pred = model.predict(X_test_final)

# print("Final Predection: ")
# print(y_pred[:10])


comparision = pd.DataFrame({
    "Actual predected: ": y_test,
    "model_predected: ": y_pred
}
)

# print(comparision.head(10))

accuracy = accuracy_score(y_test, y_pred)

# print(accuracy*100)


cm = confusion_matrix(y_test, y_pred)

# print("Confusion Matrix:")
# print(cm)


dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train_final, y_train)

dt_model_pred = dt_model.predict(X_test_final)

# print("Final predected dt_model: ")
# print(dt_model_pred[:10])

dt_accuracy = accuracy_score(y_test, dt_model_pred)

# print("Decision Tree Accuracy:", dt_accuracy)
# print("Decision Tree Accuracy Percentage:", dt_accuracy * 100, "%")


rm_model = RandomForestClassifier()

rm_model.fit(X_train_final, y_train)

rm_predectied = rm_model.predict(X_test_final)

# print("Rm_Model Predection")
# print(rm_predectied[:10])

rm_accuracy = accuracy_score(y_test, rm_predectied)

# print("Random Forest Accuracy:", rm_accuracy)
# print("Random Forest Accuracy Percentage:", rm_accuracy * 100, "%")


# Confusion Metrics

# logistic regression


cm = confusion_matrix(y_test, y_pred)

# print("Confusion Matrix:")
# print(cm)


correct_predictions = 926 + 209
total_predictions = 1409

accuracy = correct_predictions / total_predictions

# print("Accuracy:", accuracy)
# print("Accuracy Percentage:", accuracy * 100, "%")


# Precision
tp = 209
tp_fp = 209+109

precision = tp / tp_fp

# print("Precision: ",precision)
# print("Precision precentage: ", precision*100, "%")


# Recall
tp = 209
tp_fn = 209 + 165

recall = tp / tp_fn

# print("Recall:", recall)
# print("Recall Percentage:", recall * 100, "%")


# F1_score

precision = 0.6572327044025157
recall = 0.5588235294117647

F1_score = 2 * (precision*recall)/(precision+recall)

# print("F1_Score: ",F1_score)
# print("F1_Score Precentage: ",F1_score*100,"%")


report = classification_report(y_test, y_pred)

# print("Classification report: ")
# print(report)


# ROC


y_prob = model.predict_proba(X_test_final)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label="Yes")

plt.plot(fpr, tpr, label="Logistic Regression")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
# plt.show()


auc_score = roc_auc_score(y_test, y_prob)
# print("ROC-AUC Score:", auc_score)


# Model Comparision

# Logistic Regression Accuracy
log_accuracy = accuracy_score(y_test, y_pred)

# Decision Tree Accuracy
dt_accuracy = accuracy_score(y_test, dt_model_pred)

# Random Forest Accuracy
rf_accuracy = accuracy_score(y_test, rm_predectied)

# print("Logistic Regression:", log_accuracy * 100, "%")
# print("Decision Tree:", dt_accuracy * 100, "%")
# print("Random Forest:", rf_accuracy * 100, "%")


log_precision = precision_score(y_test, y_pred, pos_label="Yes")
log_recall = recall_score(y_test, y_pred, pos_label="Yes")
log_f1 = f1_score(y_test, y_pred, pos_label="Yes")

# print("Logistic Regression")
# print("Precision:", log_precision * 100, "%")
# print("Recall:", log_recall * 100, "%")
# print("F1-Score:", log_f1 * 100, "%")


rm_precision = precision_score(y_test, rm_predectied, pos_label="Yes")
rm_recall = recall_score(y_test, rm_predectied, pos_label="Yes")
rm_f1 = f1_score(y_test, rm_predectied, pos_label="Yes")

# print("Random Forest")
# print("Precision:", rm_precision * 100, "%")
# print("Recall:", rm_recall * 100, "%")
# print("F1-Score:", rm_f1 * 100, "%")


dt_precision = precision_score(y_test, dt_model_pred, pos_label="Yes")
dt_recall = recall_score(y_test, dt_model_pred, pos_label="Yes")
dt_f1 = f1_score(y_test, dt_model_pred, pos_label="Yes")

# print("Decision Tree")
# print("Precision:", dt_precision * 100, "%")
# print("Recall:", dt_recall * 100, "%")
# print("F1-Score:", dt_f1 * 100, "%")


# Logistic Regression
log_probability = model.predict_proba(X_test_final)[:, 1]
log_auc = roc_auc_score(y_test, log_probability)

# Decision Tree
dt_probability = dt_model.predict_proba(X_test_final)[:, 1]
dt_auc = roc_auc_score(y_test, dt_probability)

# Random Forest
rf_probability = rm_model.predict_proba(X_test_final)[:, 1]
rf_auc = roc_auc_score(y_test, rf_probability)

# print("Logistic Regression ROC-AUC:", log_auc)
# print("Decision Tree ROC-AUC:", dt_auc)
# print("Random Forest ROC-AUC:", rf_auc)


# Overfiting

log_train_pred = model.predict(X_train_final)
log_test_pred = model.predict(X_test_final)

log_train_accuracy = accuracy_score(y_train, log_train_pred)
log_test_accuracy = accuracy_score(y_test, log_test_pred)

# print("Logistic Regression")

# print("Training Accuracy:", log_train_accuracy * 100, "%")
# print("Testing Accuracy:", log_test_accuracy * 100, "%")

# Training predictions
dt_train_pred = dt_model.predict(X_train_final)

# Testing predictions
dt_test_pred = dt_model.predict(X_test_final)

# Accuracy calculate karo
dt_train_accuracy = accuracy_score(y_train, dt_train_pred)
dt_test_accuracy = accuracy_score(y_test, dt_test_pred)

# print("Decision Tree")

# print("Training Accuracy:", dt_train_accuracy * 100, "%")
# print("Testing Accuracy:", dt_test_accuracy * 100, "%")


rm_train_pred = rm_model.predict(X_train_final)
rm_test_pred = rm_model.predict(X_test_final)

rm_train_accuracy = accuracy_score(y_train, rm_train_pred)
rm_test_accuracy = accuracy_score(y_test, rm_test_pred)

# print("Random Forest")

# print("Training Accuracy", rm_train_accuracy * 100, "%")
# print("Testing Accuracy:", rm_test_accuracy * 100, "%")


# Cross Validation


cv_model = LogisticRegression()

scores = cross_val_score(
    cv_model,
    X_train_final,
    y_train,
    cv=5,
    scoring="accuracy"
)

# print("Cross Validation Scores:", scores)
# print("Average Accuracy:", scores.mean() * 100, "%")


# Hyperparameters


dt_model = DecisionTreeClassifier(random_state=42)

param_grid = {
    "max_depth": [3, 5, 7],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    estimator=dt_model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy"
)

grid_search.fit(X_train_final, y_train)

# print("Best Parameters:", grid_search.best_params_)
# print("Best CV Accuracy:", grid_search.best_score_ * 100, "%")

best_dt_model = grid_search.best_estimator_

best_dt_pred = best_dt_model.predict(X_test_final)


best_dt_accuracy = accuracy_score(y_test, best_dt_pred)

# print("Tuned Decision Tree Accuracy:", best_dt_accuracy * 100, "%")


param_distributions = {
    "max_depth": [3, 5, 7, 10],
    "min_samples_split": [2, 5, 10, 15]
}

random_search = RandomizedSearchCV(
    estimator=dt_model,
    param_distributions=param_distributions,
    n_iter=5,
    cv=5,
    scoring="accuracy",
    random_state=42
)

random_search.fit(X_train_final, y_train)

# print("Best Parameters:", random_search.best_params_)

# print("Best CV Accuracy:", random_search.best_score_ * 100, "%")

best_random_model = random_search.best_estimator_
# print(best_random_model)

best_random_pred = best_random_model.predict(X_test_final)

random_test_accuracy = accuracy_score(y_test, best_random_pred)

# print("RandomizedSearchCV Test Accuracy:", random_test_accuracy * 100, "%")


# Our selected model
# Logistic Regression


final_model = LogisticRegression()

final_model.fit(X_train_final, y_train)

final_predictions = final_model.predict(X_test_final)

# print(final_predictions[:10])

final_model_accuracy = accuracy_score(y_test, final_predictions)

# print("Final_model_Accuracy:")
# print(final_model_accuracy*100, "%")


# joblib.dump(final_model, "customer_churn_model.pkl")

loaded_model = joblib.load("customer_churn_model.pkl")

# print("Model loaded successfully!")

loaded_predictions = loaded_model.predict(X_test_final)

print("Loaded Model Predictions:")
print(loaded_predictions[:10])
