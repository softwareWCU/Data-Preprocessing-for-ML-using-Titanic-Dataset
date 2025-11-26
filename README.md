
# 📘 Student Performance Dataset – Analysis & Modeling

This repository contains a dataset and analysis focused on understanding the factors that influence student academic performance. The dataset includes demographic information, parental background, lunch type, test preparation status, and exam scores across three subjects.

---

## 📂 **Dataset Description**

The dataset contains the following variables:

| Feature                | Description                                  |
| ---------------------- | -------------------------------------------- |
| **Gender**             | Student gender (Male/Female)                 |
| **Ethnicity**          | Student’s ethnic group (A–E)                 |
| **Parental Education** | Highest education level achieved by parents  |
| **Lunch Type**         | Standard or Free/Reduced lunch               |
| **Test Preparation**   | Completion status of test preparation course |
| **Math Score**         | Student’s mathematics exam score             |
| **Reading Score**      | Student’s reading exam score                 |
| **Writing Score**      | Student’s writing exam score                 |

This dataset is widely used in **educational analytics**, **behavioral studies**, and **machine learning model training** for performance prediction.

---

## 🎯 **Project Objectives**

* Explore how student background variables affect exam performance
* Perform data cleaning and preprocessing
* Visualize distributions and correlations


---

## 🧪 **Files Included**

* **student_performance.csv** – Raw dataset
* **student_analysis.ipynb** – Full analysis in Google Colab
* **README.md** – Project documentation

---

## 📊 **Analysis Steps in Google Colab**

The notebook includes:

* Importing dataset using Google Colab
* Handling missing values and data cleaning
* Label encoding and feature scaling
* Exploratory Data Analysis (EDA)
* Visualization using Matplotlib

---

## 🚀 **How to Run in Google Colab**

1. Open the `.ipynb` notebook directly in Google Colab
2. Upload the dataset to Colab or load it from GitHub
3. Run each cell step-by-step
4. Modify or extend the analysis as needed

If loading from GitHub, use:

```python
import pandas as pd

url = "https://raw.githubusercontent.com/feysel2003/ML-Student-Performance-Prediction-Using-Decision-Trees/main/studentperformance.csv"
df = pd.read_csv(url)
df.head()
```

---

## 📈 **Possible Improvements**

* Try multiple ML algorithms and compare results
* Add hyperparameter tuning using GridSearchCV
* Build dashboards using Streamlit
* Analyze fairness and bias across demographic groups

---

## 🤝 **Contributors**

* **Instructor: Mesay Aschalew, MSc**
* **Feysel Mifta**


## 📜 **License**

This project is for educational and academic use.

