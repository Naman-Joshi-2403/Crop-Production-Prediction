# 🌾 Crop Production Prediction

## 📌 Project Overview

This project focuses on predicting crop production (in tons) using
agricultural data from FAOSTAT. The objective is to build regression
models that forecast production based on:

-   Area Harvested (hectares)
-   Yield (kg per hectare)
-   Year
-   Crop Type
-   Region

The final solution includes data cleaning, exploratory data analysis
(EDA), machine learning model building, and a Streamlit web application
for real-time predictions.

------------------------------------------------------------------------

## 🏗️ Project Architecture

    CROP-PRODUCTION-PREDICTION
    │
    ├── Code
    │   ├── app.py
    │   ├── data_cleaning.py
    │   ├── EDA_Dashboard.py
    │
    ├── Input
    │
    ├── Model Training
    │   ├── crop-predication-linear-regression.ipynb
    │   ├── crop-production-xg-boost.ipynb
    │
    ├── Models
    │   ├── crop_production_linear_regression.pkl
    │   ├── crop_production_xgb.pkl
    │
    ├── Output
    │   ├── Model_dataset.csv
    │
    ├── Crop_Production_Prediction.pdf
    ├── README.md
    └── dev.env

------------------------------------------------------------------------

## 📊 Business Use Cases

-   🌍 Food security and planning
-   📈 Market price forecasting
-   🚜 Agricultural policy development
-   📦 Supply chain optimization
-   🌱 Precision farming recommendations

------------------------------------------------------------------------

## 🔄 Project Workflow

### 1️⃣ Data Preparation

-   Data cleaning and preprocessing
-   Handling missing values
-   Feature engineering
-   Encoding categorical variables

### 2️⃣ Exploratory Data Analysis (EDA)

-   Crop distribution analysis
-   Year-wise production trends
-   Region-wise productivity comparison
-   Correlation analysis

### 3️⃣ Model Building

-   Linear Regression
-   XGBoost Regressor

### 4️⃣ Model Evaluation Metrics

-   R² Score
-   MAE (Mean Absolute Error)
-   MSE (Mean Squared Error)
-   RMSE

XGBoost performed better due to its ability to handle nonlinear
relationships and complex feature interactions.

------------------------------------------------------------------------

## 🚀 Streamlit Application

The Streamlit app allows users to:

-   Select Region
-   Select Crop
-   Enter Year
-   Enter Area Harvested
-   Enter Yield

And get real-time crop production predictions.

### Run the App

``` bash
streamlit run Code/app.py
```

------------------------------------------------------------------------

## ⚙️ Technologies Used

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   XGBoost
-   Streamlit
-   Matplotlib / Seaborn
-   FAOSTAT Dataset

------------------------------------------------------------------------

## 📁 Dataset

Source: FAOSTAT Agricultural Dataset

Key Features: - Area - Item (Crop) - Year - Area Harvested (ha) - Yield
(kg/ha) - Production (tons)

------------------------------------------------------------------------

## 📈 Model Comparison

  Model               Strengths
  ------------------- ----------------------------------------
  Linear Regression   Simple, interpretable
  XGBoost             Handles non-linearity, higher accuracy

Recommended Model: **XGBoost Regressor**

------------------------------------------------------------------------

## 📌 How to Use This Repository

1.  Clone the repository
2.  Install dependencies
3.  Run data cleaning script
4.  Train model (optional)
5.  Launch Streamlit app

------------------------------------------------------------------------

## 👨‍💻 Author

Naman Joshi

------------------------------------------------------------------------

## 📜 License

This project is for educational and academic purposes.
