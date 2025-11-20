# 🧠 Data Preprocessing and KNN Model using the Titanic Dataset

## 👨‍💻 Author
**Feysel Mifta**

---

## 📘 Overview
This project demonstrates a full **data preprocessing pipeline** and a **K-Nearest Neighbors (KNN)** machine learning model using the famous **Titanic dataset**.  
It focuses on transforming raw data into clean, model-ready features for predictive analysis.

---

## 🧹 Data Preprocessing Steps
The following preprocessing steps were performed in the notebook:

1. **Handling Missing Values**
   - Filled missing values in `Age`, `Embarked`, and `Fare` columns.
2. **Encoding Categorical Features**
   - Converted categorical variables (`Sex`, `Embarked`) into numeric form using label encoding or one-hot encoding.
3. **Feature Scaling**
   - Normalized numeric features to improve KNN performance.
4. **Feature Engineering**
   - Created new features such as `FamilySize` and `IsAlone` for better predictive power.
5. **Splitting Data**
   - Divided the dataset into training and testing sets (e.g., 80/20 split).

---

## 🤖 K-Nearest Neighbors (KNN) Model
After preprocessing, a **KNN classifier** was trained to predict **survival** on the Titanic dataset.

### Model Details
- **Algorithm:** K-Nearest Neighbors (KNN)
- **Library Used:** `scikit-learn`

### Example Code Snippet
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
