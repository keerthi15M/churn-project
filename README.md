# 📊 Customer Churn Prediction Project

**Author:** Keerthi M  
**GitHub:** [keerthi15M](https://github.com/keerthi15M)  
**Email:** keerthi1052031@gmail.com  
**Last Updated:** <17/11/2025>

---
## 🔎 Project Overview  
Understanding when customers might churn is crucial for subscription-based and services companies. 
In this project, we build and deploy a machine-learning pipeline that **predicts customer churn** using demographic and service usage data. 
A user-friendly web app built with Streamlit allows real-time predictions from user input.

---
## 🎯 Objectives  
- Conduct exploratory data analysis (EDA) to discover patterns and variables related to churn.  
- Preprocess data, perform feature engineering, and handle missing/irregular values.  
- Train several machine-learning models (Logistic Regression, Random Forest, XGBoost) and select the best performing.  
- Store the trained model and column metadata for production use.  
- Deploy a web application using Streamlit allowing business users to input customer details and view churn probability.  
- Provide actionable insights for business teams to enact customer retention strategies.

---
## 📂 Project Structure  
churn-project/
├── app.py                    ← Streamlit web app
├── train_model.py            ← Script to train model and save artifacts
├── requirements.txt          ← Required Python libraries
├── README.md                 ← Project documentation
├── data/                     ← Raw dataset
│   └── Telco-Customer-Churn.csv
├── models/                   ← Saved model and metadata
│   ├── xgb_churn_model.pkl
│   └── training_columns.pkl
├── notebooks/                ← EDA & model development
│   └── 01_data_exploration.ipynb
├── .gitignore                ← Files excluded from repo
└── venv/                     ← Virtual environment directory (ignored in Git)

---
## 📊 Dataset Details  
**Source:** [Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn) _(or your dataset if used)_  
**Key fields included:**  
- `CustomerID`  
- `Gender`  
- `SeniorCitizen`  
- `Partner`  
- `Dependents`  
- `Tenure`  
- `PhoneService`  
- `MonthlyCharges`  
- `TotalCharges`  
- `Churn` (target: Yes/No)

---
## 🧠 Data Processing & Modeling  
1. **Data cleaning & preprocessing**  
   - Converted `TotalCharges` to numeric (handling blank strings)  
   - Dropped `CustomerID` as not predictive  
   - One-hot encoded categorical variables using `pd.get_dummies()`  
   - Mapped target `Churn` to binary (`Yes`→1, `No`→0)  
2. **Train/Test Split**  
   - Stratified split to maintain churn ratio  
3. **Model training**  
   - Tried Logistic Regression, Random Forest, and XGBoost  
   - Evaluated using accuracy, precision, recall, F1-score, ROC-AUC  
   - Chose **XGBoost** for best performance  
4. **Model artifacts**  
   - `xgb_churn_model.pkl` → Trained XGBoost model  
   - `training_columns.pkl` → List of column names used for prediction (ensures correct order/features in the app)  

---
## 🖥️ Streamlit Web App  
The web app (`app.py`) allows users to enter new customer details and get a churn prediction.  

### How to run the app locally:  
```bash
# Navigate to project folder
cd churn-project

# Activate your virtual environment (Windows example)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
Then open http://localhost:8501 in your browser to view the dashboard.

# Features:
Two-column form layout for customer inputs
Example preset button for “High Risk of Churn” scenario
Real-time delivery of churn prediction (yes/no) and probability
“See input used for prediction” expander to view what was entered
Responsive design and user-friendly interface
```

## 📊 Key Business Insights
Based on the developed model and analysis, some significant findings include:
- Customers with short tenure (newer customers) are more likely to churn.
- Monthly charges and TotalCharges show a relationship: higher charges often correlate with churn.
- Customers using month-to-month contracts and paperless billing show higher churn risk.
- Senior citizens and those without partners/dependents show different churn behavior (should be interpreted carefully).
- The model gives business teams a probability score, helping them prioritise retention efforts.
- These insights should be updated to match your EDA results and model findings.

## 📋 Requirements
Your environment should include the following (versions may vary slightly):
- pandas==2.3.3
- numpy==2.3.4
- scikit-learn==1.7.2
- xgboost==2.1.2
- streamlit==1.40.1
- joblib==1.5.2
- matplotlib==3.10.7
- seaborn==0.13.2

Install with:
- pip install -r requirements.txt

## 📌 Important Notes
- Ensure you run streamlit run app.py from the root of the project (where models/ folder exists).
- If you used a different training-script name or dataset path, update paths accordingly.
- If the model files grow larger, consider using Git LFS
 or storing them externally and using a download link.

## 🙌 Acknowledgements
- Data science & machine learning community
- Open-source libraries (pandas, scikit-learn, xgboost, streamlit)
Internship program at Codec Technologies (or your host organisation)

## 📬 Contact
Keerthi — keerthi15M
- 📧 keerthi1052031@gmail.com

Feel free to connect if you have any questions, want to collaborate, or discuss data-science projects!
