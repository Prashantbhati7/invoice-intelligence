# 🚀 Vendor Invoice Intelligence System

## 📌 Table of Contents
- [Project Overview](#project-overview)
- [Business Objective](#business-objective)
- [Data Source](#data-source)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Application](#application)
- [Project Structure](#project-structure)
- [How to Run This Project](#how-to-run-this-project)
- [Author](#author)

---

## 📖 Project Overview

This project implements an end-to-end Machine Learning system designed to support finance teams by:

- Predicting expected freight costs for vendor invoices  
- Flagging high-risk invoices that require manual review  

The system helps detect abnormal cost, freight, and operational patterns.

---

## 🎯 Business Objective

- **Invoice Cost Leakage:** Minimize unnecessary expenses  
- **Audit Risk Management:** Ensure compliance and financial accuracy  

---

## 🗄️ Data Source

Data is stored in a relational SQLite database containing:

- Vendor Invoice Table  
- Purchase Table  
- Purchase Prices Table  
- Inventory Table  

---

## 📊 Exploratory Data Analysis (EDA)

- Analyzed cost distribution and anomaly patterns  
- Performed feature engineering  
- Used statistical tests (T-test) to validate feature significance  

---

## 🤖 Models Used

### Regression (Freight Cost)
- Linear Regression  
- Decision Tree  
- Random Forest  

### Classification (Invoice Risk)
- Logistic Regression  
- Random Forest (**final model**)  
  - Tuned using GridSearchCV  
  - Handles class imbalance  

---

## 📏 Evaluation Metrics

### Regression
- RMSE  
- R² Score  

### Classification
- Precision  
- Recall  
- F1 Score  

---

## 💻 Application

- Built using **Streamlit**
- Provides a user-friendly interface for:
  - Inputting invoice data  
  - Getting real-time predictions  
  - Flagging risky invoices  

---

## 📁 Project Structure

```
├── 📁 freight_cost_prediction
│   ├── 📁 models
│   ├── data_preprocessing.py
│   ├── model_evaluation.py
│   └── train.py
├── 📁 invoice_flagging
│   ├── data_preprocessing.py
│   ├── model_evaluation.py
│   └── train.py
├── 📁 inferencing
│   ├── predict_freight.py
│   └── predict_invoice_flag.py
├── 📁 models
│   ├── predict_flag_invoice.pkl
│   ├── predict_freight_model.pkl
│   └── scaler.pkl
├── 📁 notebooks
│   ├── flagVendorInvoice.ipynb
│   └── freightCost.ipynb
├── .gitignore
├── app.py
└── requirements.txt
```

---

## 🛠️ How to Run This Project

###  Clone the Repository

```bash
git clone https://github.com/Prashantbhati7/invoice-intelligence.git
cd invoice-intelligence
```

--
###  Train Models

#### Freight Cost Model

```bash
python freight_cost_prediction/train.py
```

#### Invoice Risk Model

```bash
python invoice_flagging/train.py
```

---

### 5. Run the Application

```bash
streamlit run app.py
```

---

## ⚠️ Notes

- Make sure `.pkl` model files are generated before running the app  
- If errors occur, verify dependencies installation  
- Use a virtual environment to avoid conflicts  

---

## 👤 Author

**Prashant Bhati**
