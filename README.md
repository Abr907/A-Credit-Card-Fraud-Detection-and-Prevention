# A-Credit-Card-Fraud-Detection-and-Prevention
# 📍 Introduction

What is the project?
This project implements a machine learning-based system for detecting and preventing credit card fraud. It involves preprocessing transactional data, handling class imbalance, and training classification models such as Logistic Regression and Random Forest. Model performance is evaluated using metrics like accuracy, precision, recall and F1-score.

Data storage:

All recipes and ingredients are stored in a PostgreSQL cloud database hosted on Neon, offering:
•	High availability
•	Strong security
•	Easy scalability

# 📍 Define the Objective

My principal goal is to detect fraudulent transactions using data and machine learning.
Problem type: Binary classification (fraud = 1, non-fraud = 0)
Objective: Build a model that predicts if a given transaction is fraudulent.

# 🔍 Usage Example
 Import libraries

import pandas as pd
import joblib
1. Load the trained model
model = joblib.load("models/fraud_model.pkl")

2. Example transaction
new_transaction = pd.DataFrame({
    'V1': [-1.3598],
    'V2': [-0.0728],
    'V3': [2.5363
    'Amount': 
    Make prediction
prediction = model.predict(new_transaction)[0]

Display result
if prediction == 1:
    print("🚨 Alert: Fraudulent transaction detected!")
else:
    print("✅ Transaction is legitimate.")
})

Make prediction
prediction = model.predict(new_transaction)[0]

 Display result
if prediction == 1:
    print("🚨 Alert: Fraudulent transaction detected!")
else:
    print("✅ Transaction is legitimate.")

# 🧠 Recommended File Highlights
A complete project overview with:
# 1. Get the Dataset
# 2. Exploratory Data Analysis (EDA)
# 3. Data Preprocessing
# 4. Visualizations
# 6. Model Evaluation
# 7. Fraud Prevention Strategies (Beyond ML)
# 8. Present the Results (metrics & plots)

# 🧩 Architecture

This is the structure :
## credit_card_fraud/
## │
## ├── data/                      Raw and processed data
## ├── notebooks/                 Jupyter notebooks for EDA and modeling
## ├── scripts/                   Data preprocessing and training scripts
## ├── models/                   Trained model files (e.g., fraud_model.pkl)
## ├── app/                      Flask/FastAPI application for deployment
## └── requirements.txt 


# 🧪 Technologies.
 
1. Programming Languages.

Python 🐍 – the primary language for data analysis, machine learning, and API development.
Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost, imbalanced-learn, tensorflow/keras.

🧹 2. Data Handling & Preprocessing.

Pandas → for data manipulation and cleaning.
NumPy → for numerical operations.
Scikit-learn → for preprocessing (scaling, encoding, feature selection).
Imbalanced-learn (SMOTE, ADASYN) → to handle class imbalance (fraudulent vs non-fraudulent transactions).

📊 3. Data Visualization

Matplotlib and Seaborn → for static EDA plots.
Plotly → for interactive dashboards and visualizations.

Power BI / Tableau (optional) → for business-level visualization and reporting.
🤖 4. Machine Learning & AI

Scikit-learn → for traditional models:
Logistic Regression
Decision Tree

Random Forest
Gradient Boosting
# 💻 Execution on Google Colab
If you prefer running my project code: 👉 Open: https://colab.research.google.com/drive/1BnG_8HiKfaZ9Yfepb1ZkJS_HjwWLtY0m?usp=sharing

Steps :

Upload the dataset (creditcard.csv) when prompted.
Run all cells sequentially.
The notebook will automatically download the /reports CSV files when finished.

## 👨‍💻 Author
Apolo Barnabas  Developers Institute — Tel Aviv/ Israel


