# ===============================
# 🚢 Titanic Survival Prediction
# Enhanced KNN Model with Feature Engineering
# ===============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set visualization style
sns.set(style='whitegrid', palette='muted')

# -----------------------------------
# 1️⃣ Load Dataset
# -----------------------------------
url = "https://raw.githubusercontent.com/softwareWCU/Data-Preprocessing-for-ML-using-Titanic-Dataset/main/titanic2.csv"
df = pd.read_csv(url)

print("Dataset Loaded ✅")
print(df.head())

# -----------------------------------
# 2️⃣ Handle Missing Values
# -----------------------------------
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

imputer = KNNImputer(n_neighbors=5)
df[['Age','Fare']] = imputer.fit_transform(df[['Age','Fare']])

df.drop(columns=['Cabin'], inplace=True, errors='ignore')
df.drop_duplicates(inplace=True)

# -----------------------------------
# 3️⃣ Clean Inconsistent Categories
# -----------------------------------
df['Sex'] = df['Sex'].str.lower().replace({'femalee':'female','mal':'male','Male':'male','Female':'female'})
df['Embarked'] = df['Embarked'].replace({'Southmpton':'S','Queenstown':'Q','Cherbourg':'C','cherbourg':'C'})
df['Embarked'] = df['Embarked'].str.lower()

# -----------------------------------
# 4️⃣ Feature Engineering
# -----------------------------------
# Extract Title
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
title_map = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Dr':5, 'Rev':6, 'Col':7, 'Major':8,
             'Mlle':2, 'Countess':3, 'Mme':3, 'Don':1, 'Lady':3, 'Sir':1, 'Jonkheer':1, 'Capt':7}
df['Title'] = df['Title'].map(title_map).fillna(0)

# Clean Pclass
df['Pclass'] = df['Pclass'].astype(str).str.replace('st','').str.replace('nd','').str.replace('rd','').str.strip().astype(int)

# New engineered features
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,20,40,60,80], labels=False)
df['FareBin'] = pd.qcut(df['Fare'], 4, labels=False)

# -----------------------------------
# 5️⃣ Encoding
# -----------------------------------
label = LabelEncoder()
df['Sex'] = label.fit_transform(df['Sex'])
df['Embarked'] = label.fit_transform(df['Embarked'])

# -----------------------------------
# 6️⃣ Feature Selection
# -----------------------------------
features = ['Pclass', 'Sex', 'AgeBin', 'FareBin', 'Embarked', 'Title', 'FamilySize', 'IsAlone']
target = 'Survived'

X = df[features]
y = df[target]

# -----------------------------------
# 7️⃣ Split and Scale
# -----------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# -----------------------------------
# 8️⃣ Optimize K using GridSearchCV
# -----------------------------------
param_grid = {'n_neighbors': list(range(1, 30)), 'weights': ['uniform', 'distance']}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

best_k = grid.best_params_['n_neighbors']
best_weight = grid.best_params_['weights']
best_acc = grid.best_score_

print(f"✅ Best K: {best_k}, Weights: {best_weight}, Cross-Validation Accuracy: {best_acc:.4f}")

# -----------------------------------
# 9️⃣ Final Model
# -----------------------------------
knn = KNeighborsClassifier(n_neighbors=best_k, weights=best_weight)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
# -----------------------------------
# 🔟 Evaluation
# -----------------------------------
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Test Data)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

acc = accuracy_score(y_test, y_pred)
print(f"🎯 Test Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))