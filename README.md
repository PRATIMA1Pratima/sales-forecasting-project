Sales Forecasting System


Overview

This project is a machine learning-based system designed to predict Walmart weekly sales using historical sales data and key business factors. It also includes a web application built with Streamlit to provide real-time predictions and a simple analytics dashboard.

Objective

The main objective of this project is to forecast retail sales based on features such as store number, holiday flag, temperature, fuel price, CPI, unemployment rate, and month. This helps in understanding sales patterns and improving business decision-making.

Technologies Used

Python, Pandas, NumPy, Scikit-learn, Streamlit, Plotly, Joblib, SQL


Project Workflow


Data Collection

The dataset used contains Walmart sales data with multiple store-level and economic variables.

Data Preprocessing

Data cleaning and preprocessing were performed. The date column was converted to datetime format and the month feature was extracted for better analysis.

Feature Selection

The important features used for training include store, holiday flag, temperature, fuel price, CPI, unemployment, and month.

Model Building

A Random Forest Regressor was used to build the prediction model.

Model Evaluation

The model achieved an R2 score of approximately 0.94 and a mean absolute error of around 71,800.


Application Features

The Streamlit application allows users to input values such as store, month, and economic indicators to predict weekly sales. It also displays key metrics and visual insights.


Project Structure

Sales_Forecasting_Project
streamlit folder contains app.py and model file
data folder contains dataset
notebooks folder contains analysis and model training
sql folder contains SQL queries
requirements.txt contains dependencies
README.md contains project description


Deployment

The project is deployed using Streamlit Cloud and can be accessed through a web browser for real-time predictions.


Future Improvements

The model can be further improved using advanced algorithms such as XGBoost or LSTM. Integration with Power BI and better time series forecasting techniques can also enhance performance.


Author

Pratima
B.Tech CSE (AI and Machine Learning)