# 💹 Price Index Predictor

A Streamlit app to explore product price data, train regression models (Decision Tree), and predict average price spreads for products.
The app includes interactive predictions, hyperparameter tuning, and data visualizations.

## 📌 Project Overview

This project is built with Python and Streamlit. It allows users to:

Explore product price data from a CSV file (ProductPriceIndex.csv).

Train a Decision Tree Regressor to predict average spreads.

Tune hyperparameters using RandomizedSearchCV.

Evaluate model performance using R², MAE, and MSE.

Visualize correlations, pairplots, and top products.

Predict average spreads for new product inputs interactively.

## ⚡ Features
### 1. Data Exploration

View raw data.

Dataset summary statistics.

Check missing values.

### 2. Feature Engineering

Convert date column to year, month, day.

Convert numeric columns from strings to floats ($ and % removed).

Encode categorical productname column using LabelEncoder.

### 3. Model Training

Train Decision Tree Regressor with RandomizedSearchCV.

Interactive hyperparameter tuning from the sidebar:

max_depth, min_samples_split, criterion, and more.

Display best hyperparameters and cross-validation score.

### 4. Predictions

Predict average spread for a new product interactively.

Enter custom numeric values for product prices.

Select product from existing list.

### 5. Model Evaluation

R² Score

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

Actual vs Predicted scatter plot.

### 6. Visualizations

Correlation Heatmap

Pairplot

Top 10 products by average spread bar chart

## 🖥 Usage

Upload your CSV file (ProductPriceIndex.csv) or use the default dataset.

Explore the raw data and statistics.

Tune hyperparameters in the sidebar.

Train the Decision Tree Regressor.

Check evaluation metrics and plots.

Enter values for a new product to predict its average spread.

Visualize correlations, pairplots, and top product

## 🔧 Dependencies

Python 3.10+

streamlit

pandas

matplotlib

seaborn

scikit-learn

numpy
