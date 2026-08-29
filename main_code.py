!pip install pandas
!pip install scikit-learn
!pip install seaborn
!pip install matplotlib
!pip install numpy
!pip install imbalanced-learn
!pip install xgboost

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
except ImportError:
    print("Please install imbalanced-learn: !pip install imbalanced-learn")

try:
    from xgboost import XGBClassifier
    has_xgboost = True
except ImportError:
    print("XGBoost not found. (!pip install xgboost)")
    has_xgboost = False

%matplotlib inline

filepath = 'dataset/creditcard.csv'
if not os.path.exists(filepath):
    print(f"Error: Dataset not found at {filepath}")
else:
    df = pd.read_csv(filepath)
    print("Data loaded successfully. Shape:", df.shape)
    display(df.head())

# 1. Class distribution plot
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Class')
plt.title('Class Distribution (0 = Legit, 1 = Fraud)')
plt.yscale('log')
plt.ylabel('Count (log scale)')
plt.tight_layout()
plt.show()

# 2. Distribution of Amount and Time split by class
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df[df['Class'] == 0]['Time'], bins=50, color='blue', alpha=0.5, label='Legit', ax=axes[0], stat='density')
sns.histplot(df[df['Class'] == 1]['Time'], bins=50, color='red', alpha=0.5, label='Fraud', ax=axes[0], stat='density')
axes[0].set_title('Density of Time by Class')
axes[0].legend()

sns.histplot(df[df['Class'] == 0]['Amount'], bins=50, color='blue', alpha=0.5, label='Legit', ax=axes[1], stat='density')
sns.histplot(df[df['Class'] == 1]['Amount'], bins=50, color='red', alpha=0.5, label='Fraud', ax=axes[1], stat='density')
axes[1].set_title('Density of Amount by Class')
axes[1].set_yscale('log')
axes[1].legend()
plt.tight_layout()
plt.show()

# 3. Correlation heatmap across V1-V28
print("Generating correlation heatmap (this may take a moment)...")
plt.figure(figsize=(20, 16))
v_features = [f'V{i}' for i in range(1, 29)] + ['Class']
corr = df[v_features].corr()
sns.heatmap(corr, cmap='coolwarm_r', center=0, annot=False)
plt.title('Correlation Heatmap (V1-V28 and Class)')
plt.tight_layout()
plt.show()

# Scale Amount and Time (V features are already PCA-scaled)
scaler = StandardScaler()
df['Scaled_Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['Scaled_Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

df.drop(['Time', 'Amount'], axis=1, inplace=True)

X = df.drop('Class', axis=1)
y = df['Class']

# Train/test split with stratify
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

print("--- Comparing Imbalance Strategies using Logistic Regression ---")

strategies = {
    'Baseline (No Resampling)': (X_train, y_train, LogisticRegression(max_iter=1000, random_state=42)),
    'Class Weight Balanced': (X_train, y_train, LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')),
    'Random Undersampling': (*RandomUnderSampler(random_state=42).fit_resample(X_train, y_train), LogisticRegression(max_iter=1000, random_state=42)),
    'SMOTE': (*SMOTE(random_state=42).fit_resample(X_train, y_train), LogisticRegression(max_iter=1000, random_state=42))
}

strategy_results = []

for name, (X_tr, y_tr, model) in strategies.items():
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    strategy_results.append({
        'Strategy': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    })
    
df_strategies = pd.DataFrame(strategy_results)
display(df_strategies)

print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=10),
    'Naive Bayes': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier(n_jobs=-1) 
}

if has_xgboost:
    models['XGBoost'] = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    
results = []
roc_curves = {}
pr_curves = {}

for name, model in models.items():
    print(f"Training {name} (This may take a while for KNN/Ensembles)...")
    model.fit(X_train_sm, y_train_sm)
    y_pred = model.predict(X_test)
    
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred
        
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall_vals, precision_vals)
    
    roc_curves[name] = (fpr, tpr, roc_auc)
    pr_curves[name] = (precision_vals, recall_vals, pr_auc)
    
    results.append({
        'Algorithm': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc
    })
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

df_results = pd.DataFrame(results)
display(df_results)

# Bar chart comparing accuracy vs F1/recall
df_melted = df_results.melt(id_vars='Algorithm', value_vars=['Accuracy', 'Recall', 'F1-Score'], 
                            var_name='Metric', value_name='Score')

plt.figure(figsize=(12, 6))
sns.barplot(data=df_melted, x='Algorithm', y='Score', hue='Metric')
plt.title('Algorithm Comparison: Accuracy vs Recall vs F1-Score')
plt.ylim(0, 1.1)
plt.xticks(rotation=45)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# ROC Curves
plt.figure(figsize=(10, 8))
for name, (fpr, tpr, roc_auc) in roc_curves.items():
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves (Using SMOTE)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# PR Curves
plt.figure(figsize=(10, 8))
for name, (prec, rec, pr_auc) in pr_curves.items():
    plt.plot(rec, prec, label=f'{name} (AUC = {pr_auc:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves (Using SMOTE)')
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

