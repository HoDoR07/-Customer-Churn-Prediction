
# Customer Churn Prediction

## Project Overview

Customer Churn Prediction is a Machine Learning project that predicts whether a customer is likely to leave a telecom service or continue using it.

The project uses customer information such as tenure, monthly charges, contract type, payment method, and subscribed services to predict customer churn.

The main objective is to help businesses identify customers who may leave their services and take steps to improve customer retention.

## Dataset Information

- **Dataset:** IBM Telco Customer Churn
- **Total Records:** 7,043
- **Original Columns:** 21
- **Target Column:** Churn
- **Target Classes:** Yes and No

The dataset contains customer details, subscribed services, contract information, payment methods, tenure, and monthly charges.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- VS Code

## Project Workflow

1. Dataset Loading
2. Dataset Understanding
3. Exploratory Data Analysis (EDA)
4. Data Cleaning
5. Data Preprocessing
6. Train-Test Split
7. Model Training
8. Model Evaluation
9. Model Comparison
10. Cross Validation
11. Hyperparameter Tuning
12. Final Model Selection
13. Model Saving and Loading

## Data Preprocessing

The following preprocessing steps were performed:

- Identified and handled blank values in TotalCharges.
- Converted TotalCharges into a numerical data type.
- Removed the customerID column because it is an identifier.
- Separated features (X) and target (y).
- Split the dataset into training and testing sets.
- Applied StandardScaler to numerical features.
- Applied One-Hot Encoding to categorical features.
- Prepared numerical datasets for model training.

### Train-Test Split

- Training Records: 5,634
- Testing Records: 1,409
- Test Size: 20%
- Random State: 42

## Machine Learning Models

Three classification algorithms were trained and compared:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

Logistic Regression was selected as the final model based on its performance on the test dataset.

## Model Performance Comparison

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 80.55% |
| Decision Tree | 72.11% |
| Random Forest | 78.64% |

## Logistic Regression Evaluation

The final Logistic Regression model achieved the following results:

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 80.55% |
| Precision | 65.72% |
| Recall | 55.88% |
| F1-Score | 60.40% |
| ROC-AUC | 0.8421 |

### Confusion Matrix

```text
[[926 109]
 [165 209]]
```

### Classification Report

```text
              precision    recall  f1-score   support

          No       0.85      0.89      0.87      1035
         Yes       0.66      0.56      0.60       374

    accuracy                           0.81      1409
   macro avg       0.75      0.73      0.74      1409
weighted avg       0.80      0.81      0.80      1409
```

## Cross Validation and Hyperparameter Tuning

- Applied 5-Fold Cross Validation.
- Used GridSearchCV for hyperparameter tuning.
- Used RandomizedSearchCV for hyperparameter tuning.
- Compared tuned Decision Tree performance.

### Best Decision Tree Parameters

```python
{
    "max_depth": 3,
    "min_samples_split": 2
}
```

- Best Cross Validation Accuracy: 78.79%
- Test Accuracy: 78.21%

## Overfitting Analysis

Training and testing accuracy were compared to understand model generalization.

- Logistic Regression showed a very small difference between training and testing accuracy.
- Decision Tree showed a large difference, indicating overfitting.
- Random Forest also showed a noticeable difference between training and testing accuracy.

## Model Saving and Loading

The final trained Logistic Regression model was saved using Joblib.

```python
import joblib

joblib.dump(final_model, "customer_churn_model.pkl")
```

The saved model was successfully loaded and used to generate predictions.

```python
loaded_model = joblib.load("customer_churn_model.pkl")

predictions = loaded_model.predict(X_test_final)
```

## Project Structure

```text
Customer_Churn_Prediction/
│
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── cleaned_customer_churn.csv
│
├── customer_churn_prediction.py
├── freshcode.py
├── customer_churn_model.pkl
├── requirements.txt
└── README.md
```

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/HoDoR07/-Customer-Churn-Prediction.git
```

### 2. Open the Project Folder

```bash
cd -Customer-Churn-Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Project

```bash
python freshcode.py
```

## Key Learnings

Through this project, I learned:

- Understanding and exploring real-world datasets.
- Performing Exploratory Data Analysis.
- Handling missing and blank values.
- Encoding categorical features.
- Scaling numerical features.
- Training classification models.
- Evaluating ML models using multiple metrics.
- Understanding overfitting.
- Applying Cross Validation.
- Performing hyperparameter tuning.
- Saving and loading trained ML models using Joblib.

## Future Improvements

- Improve churn detection and recall.
- Experiment with additional classification algorithms.
- Build a user interface for customer churn prediction.
- Integrate the trained model with a backend API.
- Deploy the application.

## Author

Raushan Kumar

Machine Learning Project – Customer Churn Prediction
