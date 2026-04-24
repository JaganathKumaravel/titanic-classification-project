# 🚢 Titanic Survival Prediction – End-to-End ML Project

## 📌 Project Overview
This project predicts whether a passenger survived the Titanic disaster using Machine Learning. It is built as an **end-to-end system**, including data preprocessing, model training, API deployment, and a web interface.

---

## 🎯 Problem Statement
Given passenger details such as age, gender, ticket class, and family size, predict whether the passenger survived (binary classification: 0 = Did Not Survive, 1 = Survived).

---

## 📊 Dataset
- Source: Kaggle Titanic Dataset
- Features used:
  - Pclass
  - Sex
  - Age
  - SibSp
  - Parch
  - Fare
  - Embarked
  - family_size (engineered feature)

---

## 🧹 Data Preprocessing
- Removed irrelevant columns: PassengerId, Name, Ticket, Cabin
- Handled missing values:
  - Age → median
  - Embarked → mode
- Feature engineering:
  - family_size = SibSp + Parch
- One-hot encoding for categorical variables

---

## 🤖 Machine Learning Model
- Algorithm: Logistic Regression
- Implemented using Scikit-learn
- Training approach: Pipeline (preprocessing + model combined)

---

## 📈 Evaluation
- Accuracy and classification report used for evaluation
- Model performs binary classification (Survived / Not Survived)

---

## 🌐 Deployment Architecture

- Backend API: FastAPI
- Frontend UI: Streamlit (local)
- Cloud Deployment: Render

Live API:
https://titanic-api-80k7.onrender.com/docs

---

## 🚀 How to Run Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
