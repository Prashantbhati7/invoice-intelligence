Vendor Invoice Intelligence System
Table of Contents

Project Overview
Business Objective
Data Source
Exploratory Data Analysis (EDA)
Models Used
Evaluation Metrics
Application
Project Structure
How to Run This Project
Author & Contact


1. Project Overview

This project implements an end-to-end Machine Learning system designed to support finance teams by predicting expected freight costs for vendor invoices and flagging high-risk invoices that require manual review due to abnormal cost, freight, or operational patterns.


2. Business Objective

Invoice Cost Leakage: Reduce unnecessary costs.
Audit Risk Management: Ensure compliance and accuracy.


3. Data Source
Data is stored in a relational SQLite database containing the following tables:
Vendor Invoice Table
Purchase Table
Purchase Prices Table
Inventory Table

5. Exploratory Data Analysis (EDA)
Analyzed cost patterns and risk behavior.
Performed feature engineering and T-test to validate the significance of variables.

5. Models Used

Regression (Freight Cost): Linear Regression, Decision Tree, Random Forest.
Classification (Invoice Risk): Logistic Regression, Random Forest (final model with GridSearchCV and hyperparameter tuning to handle class imbalance).
6. Evaluation Metrics

Regression: RMSE, R² Score.
Classification: Precision, Recall, F1 Score.
7. Application

Built with Streamlit for a modern UI allowing users to input data and get real-time predictions and alerts.

8. Project Structure
```
├── 📁 Freight_cost_prediction
│   ├── 📁 models
│   ├── 🐍 data_preprocessing.py
│   ├── 🐍 model_evaluation.py
│   └── 🐍 train.py
├── 📁 inferencing
│   ├── 🐍 predict_freight.py
│   └── 🐍 predict_invoice_flag.py
├── 📁 invoice_flagging
│   ├── 🐍 data_preprocessing.py
│   ├── 🐍 model_evaluation.py
│   └── 🐍 train.py
├── 📁 models
│   ├── 📄 predict_flag_invoice.pkl
│   ├── 📄 predict_freight_model.pkl
│   └── 📄 scaler.pkl
├── 📁 notebooks
│   ├── 📄 flagVendorInvoice.ipynb
│   └── 📄 freightCost.ipynb
├── ⚙️ .gitignore
├── 📄 Linear Regression.pkl
└── 🐍 app.py
```


9. How to Run This Project

## 🛠️ How to Run This Project

Follow the steps below to set up and run the project locally.

---

### 1. Prerequisites

- Python 3.8 or higher installed
- pip installed

---

### 2. Clone the Repository

```bash
git clone https://github.com/Prashantbhati7/invoice-intelligence.git
```

---

### 2. Model Training

Run the following commands:

#### Train Regression Model (Freight Cost)

```bash
python freight_cost_prediction/train.py
```

#### Train Classification Model (Invoice Risk)

```bash
python invoice_flaging/train.py
```

---

### 6. Run the Application

```bash
streamlit run app.py
```

---

### Notes

- Ensure `.pkl` model files are generated before running the app.
- If errors occur, re-check dependencies installation.
- Using a virtual environment is recommended.



10. Author

Prashant bhati

