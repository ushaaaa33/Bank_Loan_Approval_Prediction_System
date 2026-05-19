# 🏦 Bank Loan Approval Prediction

A complete end-to-end machine learning project that predicts whether a bank loan
application will be **approved** or **rejected** based on applicant details.
Built with Python, Scikit-learn, XGBoost, and Streamlit.

---

## 📁 Project Structure

```
bank-loan-approval-prediction/
│
├── loan_approval_data.csv    # Raw dataset (1000 rows, 20 columns)
├── loan.ipynb                # Complete Jupyter Notebook (EDA → Tuning)
├── app.py                    # Streamlit web application
├── loan_model.pkl            # Trained XGBoost model (saved with joblib)
├── encoders.pkl              # Label encoders for categorical features
├── feature_names.pkl         # Ordered list of features used in training
└── README.md                 # Project documentation
```

---

## 📊 Dataset Description

| Column              | Type        | Description                              |
|---------------------|-------------|------------------------------------------|
| Applicant_ID        | Float       | Unique ID (dropped before training)      |
| Applicant_Income    | Float       | Monthly income of the applicant          |
| Coapplicant_Income  | Float       | Monthly income of the co-applicant       |
| Employment_Status   | Categorical | Salaried / Self-employed / Contract / Unemployed |
| Age                 | Float       | Applicant age in years                   |
| Marital_Status      | Categorical | Married / Single                         |
| Dependents          | Float       | Number of dependents                     |
| Credit_Score        | Float       | Applicant's credit score (300–900)       |
| Existing_Loans      | Float       | Number of existing active loans          |
| DTI_Ratio           | Float       | Debt-to-Income ratio (0.0–1.0)           |
| Savings             | Float       | Total savings amount                     |
| Collateral_Value    | Float       | Value of collateral offered              |
| Loan_Amount         | Float       | Requested loan amount                    |
| Loan_Term           | Float       | Loan duration in months                  |
| Loan_Purpose        | Categorical | Personal / Home / Car / Business / Education |
| Property_Area       | Categorical | Urban / Semiurban / Rural                |
| Education_Level     | Categorical | Graduate / Not Graduate                  |
| Gender              | Categorical | Female / Male                            |
| Employer_Category   | Categorical | Private / Government / MNC / Business / Unemployed |
| Loan_Approved       | Categorical | **Target** — Yes (Approved) / No (Rejected) |

**Dataset size:** 1000 rows | **Missing values:** ~50 rows per column (imputed)

---

## 🔬 Project Steps

### 1. Data Collection & Exploration
- Loaded dataset with Pandas
- Examined shape, data types, and summary statistics
- Identified missing values (50 per column) and class distribution

### 2. Data Cleaning & Transformation
- Dropped rows with missing target (`Loan_Approved`)
- Removed `Applicant_ID` (non-predictive)
- Imputed numerical columns with **median**
- Imputed categorical columns with **mode**
- Capped outliers using **Winsorization** (IQR ×1.5)
- Applied **Label Encoding** Education_Level & Loan_Approved, **OneHotEncoder** for 6 categorical features

### 3. Exploratory Data Analysis (EDA)
- Distribution plots for all numerical features
- Correlation heatmap
- Boxplots: Credit Score and Income vs Loan Approval
- Distribution comparison: DTI Ratio and Savings by approval status

### 4. Feature Selection
- Used Random Forest feature importances
- Removed features with importance < 0.01 (if any)
- Final features retained: 27 columns

### 5. Model Training & Evaluation
Trained and compared 7 models:

| Model                | Test Accuracy |
|----------------------|---------------|
| Logistic Regression  | ~86%          |
| Decision Tree        | ~89%          |
| Random Forest        | ~91%          |
| KNN                  | ~75%          |
| Naive Bayes          | ~86%          |
| Gradient Boosting    | ~91%  
| Bagging                ~91%
  AdaBoost               ~90%
| **XGBoost** ✅       | **~92%**      |

### 6. Hyperparameter Tuning (XGBoost — GridSearchCV)
```
Best Parameters:
  n_estimators     : 100
  max_depth        : 5
  learning_rate    : 0.01
  subsample        : 1.0
  colsample_bytree : 1.0

Best CV Accuracy  : 95.00%
Test Accuracy     : 92.50%
```

### 7. Model Deployment
- Built an interactive Streamlit web app on localhost
- Users fill in applicant details and get instant predictions with confidence scores

---

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit joblib
```

### Step 1 — Run the Notebook
Open `loan.ipynb` in Jupyter and run all cells top to bottom.
This will train the model and save `loan_model.pkl`, `encoders.pkl`, `feature_names.pkl`.

```bash
jupyter notebook loan.ipynb
```

### Step 2 — Launch the Streamlit App
```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

## 🛠️ Tech Stack

| Tool           | Purpose                          |
|----------------|----------------------------------|
| Python 3.x     | Core programming language        |
| Pandas         | Data loading and manipulation    |
| NumPy          | Numerical operations             |
| Matplotlib     | Data visualisation               |
| Seaborn        | Statistical visualisation        |
| Scikit-learn   | ML models, GridSearchCV, metrics |
| XGBoost        | Final prediction model           |
| Streamlit      | Web application framework        |
| Joblib         | Model serialisation              |

---

## 📈 Model Performance

```
              precision    recall  f1-score   support

   Rejected       0.99      0.96      0.98       130
   Approved       0.92      0.98      0.95        60

   accuracy                           0.92       190
  macro avg       0.96      0.97      0.96       190
