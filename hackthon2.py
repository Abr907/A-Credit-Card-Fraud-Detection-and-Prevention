import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import files
files.upload()

df = pd.read_csv("creditcard.csv")
print(df.info())
print(df['Class'].value_counts())

sns.countplot(x='Class', data=df)
plt.title("Fraud vs Normal Transaction Count")
plt.show()

# Distribution of Tans Amouts

sns.histplot(df['Amount'], bins=100, kde=True)
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Amount[$]')
plt.ylabel('Frequency')
plt.show()
# Distribution of Trans Time

sns.histplot(df['Time'], bins=100, kde=True)
plt.title('Transaction Time Distribution')
plt.xlabel('Time (seconds since first transaction)[H]')
plt.ylabel('Frequency')
plt.show()

# Distribution of Trans Amount & Class

#sns.boxplot(x='Class', y='Amount', data=df)
#plt.title('Transaction Amount by Class')
#plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='Class', y='Amount', data=df)
plt.title('Transaction Amount vs Class')
plt.xlabel('Class (0 = Non-Fraud, 1 = Fraud)')
plt.ylabel('Transaction Amount')
plt.show()

important_features = ['V12', 'V14', 'V17']

for feature in important_features:     #visualize each anonymized feature vs.Class to see patterns.
    plt.figure(figsize=(7,4))
    sns.kdeplot(data=df[df['Class']==0][feature], label='Non-Fraud', fill=True)
    sns.kdeplot(data=df[df['Class']==1][feature], label='Fraud', fill=True, color='red')
    plt.title(f'Distribution of {feature} by Class')
    plt.legend()
    plt.show()

from sklearn.preprocessing import StandardScaler

# Copy dataframe
df_copy = df.copy()

# Standardize 'Amount' and 'Time' only
scaler = StandardScaler()
df_copy[['Amount', 'Time']] = scaler.fit_transform(df_copy[['Amount', 'Time']])

from sklearn.preprocessing import StandardScaler

# Define X and y
X = df_copy.drop('Class', axis=1)
y = df_copy['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  #Standardization

X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
print(df['Class'].value_counts())  # Before applying any balancing technique, check how imbalanced your dataset is
print(df['Class'].value_counts(normalize=True))

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

print(y_res.value_counts())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

print("Before SMOTE:", np.bincount(y_train))
print("After SMOTE:", np.bincount(y_resampled))

from imblearn.over_sampling import SMOTE


smote = SMOTE(random_state=42, sampling_strategy=0.2)
X_train_res, y_train_res = smote.fit_resample(X_train,y_train)

print("After SMOTE:", y_train_res.value_counts())
sns.countplot(x=y_train_res)
plt.title("Balanced dataset after SMOTE")
plt.show()

#  Evaluation
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    RocCurveDisplay, PrecisionRecallDisplay, fbeta_score
)

def evaluate_model(model, Xtr, ytr, Xte, yte, name=None, threshold=0.5):
    """
    Train a classification model and show key metrics + visualizations.
    - model: sklearn or XGBoost model with .fit() and .predict_proba()
    - threshold: probability cutoff for class 1 (fraud)
    """
    if name is None:
        name = model.__class__.__name__

    print(f"\n Training {name}...")
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    # Metrics
    print(f"\n=== {name} (threshold={threshold:.2f}) ===")
    print(classification_report(yte, y_pred, digits=4))
    print("ROC-AUC:", round(roc_auc_score(yte, proba), 4))

    # Metric confusion plot ===
    cm = confusion_matrix(yte, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title(f"Confusion Matrix — {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    RocCurveDisplay.from_predictions(yte, proba)
    plt.title(f"ROC Curve — {name}")
    plt.show()

    #  PRECISION-RECALL CURVE
    PrecisionRecallDisplay.from_predictions(yte, proba)
    plt.title(f"Precision-Recall — {name}")
    plt.show()

    return proba


def find_best_threshold(y_true, proba, beta=2.0):
    """
    Find the best classification threshold maximizing the F-beta score.
    beta > 1 means we value Recall more than Precision.
    """
    ts = np.linspace(0.01, 0.99, 99)
    scores = [(t, fbeta_score(y_true, (proba>=t).astype(int), beta=beta)) for t in ts]
    best_t, best_s = max(scores, key=lambda x: x[1])
    print(f" Best threshold for F{beta}: {best_t:.2f} (score={best_s:.4f})")
    return best_t


# Make sure that everything are loaded
print(" Functions 'evaluate_model()' and 'find_best_threshold()' are ready to use!")

from sklearn.linear_model import LogisticRegression

# Define baseline model
log_reg = LogisticRegression(max_iter=1000, n_jobs=-1)

# Train & evaluate
proba_lr = evaluate_model(
    log_reg,
    X_train_res, y_train_res,
    X_test, y_test,
    name="Logistic Regression"
)
# Find the optimal threshold (F2-score → Recall is more important)
t_star = find_best_threshold(y_test, proba_lr, beta=2.0)
# Re-evaluate with tuned threshold
_ = evaluate_model(
    log_reg,X_train_res, y_train_res, X_test, y_test,
    name="Logistic Regression (tuned)",
    threshold=t_star
)

# Simple summary table
from sklearn.metrics import precision_score, recall_score, f1_score

models = {
    "LogisticRegression": (log_reg, proba_lr),
    "RandomForest": (rf_model, rf_model.predict_proba(X_test)[:, 1]),

}

summary = []
for name, (model, proba) in models.items():
    y_pred = (proba >= 0.5).astype(int)
    summary.append({
        "Model": name,
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, proba)
    })

pd.DataFrame(summary).sort_values("AUC", ascending=False)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Assuming you have already run evaluate_model for a model, e.g., Logistic Regression
# and have the predicted probabilities stored in a variable like proba_lr
# If you want to evaluate a different model, replace proba_lr with the appropriate variable
proba = proba_lr # Replace with the desired model's probabilities

# Define a threshold (you can use the best threshold found earlier, e.g., t_star for Logistic Regression)
# Or you can use a default threshold of 0.5
threshold = t_star # Replace with the desired threshold

# Convert probabilities to predicted labels based on the threshold
y_pred = (proba >= threshold).astype(int)

# Basic metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"AUC-ROC: {auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style='whitegrid', palette='muted', font_scale=1.1)
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df, palette='Set2')
plt.title('Transaction Class Distribution (Before Balancing)')
plt.xticks([0,1], ['Non-Fraud (0)', 'Fraud (1)'])
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

from sklearn.ensemble import RandomForestClassifier

# Define and train a Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_res, y_train_res)

# FRAUD DETECTION DASHBOARD
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


sns.set(style="whitegrid", palette="muted", font_scale=1.1)

# ---------- 1. Dataset Overview ----------
fig, axes = plt.subplots(3, 2, figsize=(14, 16))
fig.suptitle("Credit Card Fraud Detection Dashboard", fontsize=18, fontweight='bold')

# ---------- 2. Class Distribution ----------
sns.countplot(x='Class', data=df, ax=axes[0,0], palette='coolwarm')
axes[0,0].set_title("Class Distribution (Before Balancing)")
axes[0,0].set_xticklabels(['Non-Fraud (0)', 'Fraud (1)'])
axes[0,0].set_xlabel("Class")
axes[0,0].set_ylabel("Count")
# ---------- 3. Transaction Amount ----------
sns.boxplot(x='Class', y='Amount', data=df, ax=axes[0,1], palette='Set2')
axes[0,1].set_title("Transaction Amount by Class")
axes[0,1].set_xticklabels(['Non-Fraud', 'Fraud'])
axes[0,1].set_xlabel("")
axes[0,1].set_ylabel("Transaction Amount")

# ---------- 4. Correlation Heatmap ----------
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', center=0, ax=axes[1,0], cbar=False)
axes[1,0].set_title("Feature Correlation Heatmap")

# ---------- 5. Feature Importance ----------
# Assuming rf_model is already trained in a previous cell
feat_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)[:10]
sns.barplot(x=feat_importances, y=feat_importances.index, ax=axes[1,1], palette='viridis')
axes[1,1].set_title("Top 10 Important Features (Random Forest)")
axes[1,1].set_xlabel("Importance Score")
axes[1,1].set_ylabel("Feature")

# ---------- 6. Confusion Matrix ----------
# Assuming rf_model and X_test are already defined
y_pred_rf = rf_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2,0], cbar=False)
axes[2,0].set_title("Confusion Matrix - Random Forest")
axes[2,0].set_xlabel("Predicted Label")
axes[2,0].set_ylabel("True Label")
# ---------- 7. ROC Curves ----------
# Define the models dictionary
models = {
    "Logistic Regression": log_reg, # Assuming log_reg is defined in a previous cell
    "Random Forest": rf_model # Assuming rf_model is defined and trained in a previous cell
}

# Assuming X_test, X_test_scaled, and y_test are already defined
for name, model in models.items():
    # Use X_test_scaled for models trained on scaled data, otherwise use X_test
    X_used = X_test_scaled if "Logistic" in name or "Neural" in name else X_test
    y_pred_proba = model.predict_proba(X_used)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    axes[2,1].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

axes[2,1].plot([0, 1], [0, 1], 'k--')
axes[2,1].set_title("ROC Curve Comparison")
axes[2,1].set_xlabel("False Positive Rate")
axes[2,1].set_ylabel("True Positive Rate")
axes[2,1].legend(loc='lower right')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# === FEATURE IMPORTANCE EXPORTS ===
import os
import pandas as pd

# ✅ Fix path for both Colab and Local
reports_dir = os.path.join(os.getcwd(), "reports")
os.makedirs(reports_dir, exist_ok=True)

# === Save CSV reports ===
results_path = os.path.join(reports_dir, "model_summary.csv")
rf_path = os.path.join(reports_dir, "feature_importance_rf.csv")
#xgb_path = os.path.join(reports_dir, "feature_importance_xgb.csv") # Removed as xgb_importances is not defined

# build dataframe before saving
results_df = pd.DataFrame(summary).sort_values("AUC", ascending=False)

# Convert importances to a pandas Series before saving
importances_series = pd.Series(feat_importances)

# export
results_df.to_csv(results_path, index=False)
importances_series.to_csv(rf_path, header=["importance"])
#xgb_importances.to_csv(xgb_path, header=["importance"]) # Removed as xgb_importances is not defined

print(f"📊 Reports saved to: {os.path.abspath(reports_dir)}")

# === Detect if running in Google Colab ===
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# === If Colab: download files directly ===
if IN_COLAB:
    from google.colab import files
    print("⬇️ Downloading reports to your computer...")
    files.download(results_path)
    files.download(rf_path)
    #files.download(xgb_path) # Removed as xgb_importances is not defined
else:
    print("✅ Running locally — files exported to /reports folder.")



