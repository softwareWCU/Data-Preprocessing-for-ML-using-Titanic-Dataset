# ===============================
# 🚢 Titanic Feature Engineering + Optimized KNN Model
# ===============================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------------
# 1. Load Dataset
# -----------------------------------
url = "https://raw.githubusercontent.com/softwareWCU/Data-Preprocessing-for-ML-using-Titanic-Dataset/main/titanic2.csv"
df = pd.read_csv(url)

# -----------------------------------
# 2. Handle Missing Values
# -----------------------------------
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

imputer = KNNImputer(n_neighbors=5)
df[['Age','Fare']] = imputer.fit_transform(df[['Age','Fare']])

df.drop(columns=['Cabin'], inplace=True, errors='ignore')
df.drop_duplicates(inplace=True)

# -----------------------------------
# 3. Fix Categorical Errors
# -----------------------------------
df['Sex'] = df['Sex'].str.lower().replace({'femalee':'female','mal':'male'})
df['Embarked'] = df['Embarked'].replace({
    'Southmpton':'S',
    'Queenstown':'Q',
    'Cherbourg':'C',
    'cherbourg':'C'
})
df['Embarked'] = df['Embarked'].str.lower()

# -----------------------------------
# 4. Feature Engineering
# -----------------------------------
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
title_map = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Dr':5, 'Rev':6, 'Col':7}
df['Title'] = df['Title'].map(title_map).fillna(0)

df['Pclass'] = (
    df['Pclass']
    .astype(str)
    .str.replace('st','')
    .str.replace('nd','')
    .str.replace('rd','')
    .astype(int)
)

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,20,40,60,80], labels=False)
df['FareBin'] = pd.qcut(df['Fare'], 4, labels=False)

# -----------------------------------
# 5. Encoding
# -----------------------------------
label = LabelEncoder()
df['Sex'] = label.fit_transform(df['Sex'])
df['Embarked'] = label.fit_transform(df['Embarked'])

# -----------------------------------
# 6. Select Features
# -----------------------------------
features = ['Pclass','Sex','AgeBin','FareBin','Embarked','Title','FamilySize','IsAlone']
target = 'Survived'

X = df[features]
y = df[target]

# -----------------------------------
# 7. Split and Scale
# -----------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# -----------------------------------
# 8. Optimize K using GridSearchCV
# -----------------------------------
param_grid = {
    'n_neighbors': list(range(1, 30)),
    'weights': ['uniform', 'distance']
}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

best_k = grid.best_params_['n_neighbors']
best_weight = grid.best_params_['weights']
best_accuracy = grid.best_score_

print(f"Best K: {best_k}")
print(f"Best Weight Option: {best_weight}")
print(f"Best Cross-Validation Accuracy: {best_accuracy:.4f}")

# -----------------------------------
# 9. Train Final Model
# -----------------------------------
model = KNeighborsClassifier(n_neighbors=best_k, weights=best_weight)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# -----------------------------------
# 10. Evaluation
# -----------------------------------
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\nFinal Test Accuracy: {acc:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.show()
