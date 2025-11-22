# Filename: price_index_predictor.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

st.set_page_config(page_title="💹 Price Index Predictor", layout="wide", page_icon="💹")

st.title("💹 Price Index Predictor")
st.markdown("""
Predict average product price spreads, explore the dataset, and visualize trends.
""")

# --- Load default CSV ---
@st.cache_data
def load_data():
    df = pd.read_csv("ProductPriceIndex.csv")
    return df

df = load_data()

# --- Data Overview ---
if st.checkbox("Show raw data"):
    st.dataframe(df.head())

st.subheader("Dataset Info")
st.write(df.describe())
st.write("Number of rows:", df.shape[0])
st.write("Number of columns:", df.shape[1])

# --- Feature Engineering ---
df["date"] = pd.to_datetime(df["date"], errors='coerce')
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day

# Convert numeric columns
for col, pattern in zip(
    ["averagespread", "farmprice","atlantaretail","chicagoretail","losangelesretail","newyorkretail"],
    [r'[%]', r'[\$,]', r'[\$,]', r'[\$,]', r'[\$,]', r'[\$,]']
):
    df[col] = pd.to_numeric(df[col].replace(pattern, '', regex=True), errors='coerce')

# Encode productname
le = LabelEncoder()
df["product_en"] = le.fit_transform(df["productname"].astype(str))

feature_cols = ["farmprice","atlantaretail","chicagoretail",
                "losangelesretail","newyorkretail","product_en"]
df_clean = df[feature_cols + ["averagespread"]].dropna()

x = df_clean[feature_cols]
y = df_clean["averagespread"]

# --- Split data ---
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# --- Sidebar: Model selection ---
st.sidebar.title("Model Options & Hyperparameters")
model_type = st.sidebar.selectbox("Select Model", ["Decision Tree"])

# Hyperparameters UI
if model_type == "Decision Tree":
    n_iter = st.sidebar.slider("RandomizedSearchCV iterations", 1, 20, 5)
    st.sidebar.markdown("**Decision Tree Hyperparameters:**")
    max_depth_options = st.sidebar.multiselect("Max Depth", [None,5,10,15,20,25,30], default=[None,10,20])
    min_samples_split_options = st.sidebar.multiselect("Min Samples Split", [2,5,10,20], default=[2,5])
    criterion_options = st.sidebar.multiselect("Criterion", ["squared_error","friedman_mse","absolute_error","poisson"], default=["squared_error"])

    param_grid = {
        'max_depth': max_depth_options,
        'min_samples_split': min_samples_split_options,
        'min_samples_leaf': [1,2,4,8],
        'max_features': [None, 'sqrt', 'log2'],
        'max_leaf_nodes': [None, 10, 20, 50, 100],
        'min_impurity_decrease': [0.0, 0.001, 0.01, 0.1],
        'criterion': criterion_options
    }

    st.subheader("🚀 Training Decision Tree Regressor")
    rr = RandomizedSearchCV(
        estimator=DecisionTreeRegressor(random_state=42),
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    rr.fit(x_train, y_train)
    best_dt = rr.best_estimator_

    st.success("✅ Training Complete!")
    st.write("**Best Hyperparameters:**", rr.best_params_)
    st.write("**CV Score:**", rr.best_score_)

    # Predictions & metrics
    y_pred = best_dt.predict(x_test)
    st.subheader("Model Performance")
    st.write("R² Score:", r2_score(y_test, y_pred))
    st.write("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
    st.write("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))

    # --- Actual vs Predicted Plot ---
    st.subheader("📊 Actual vs Predicted Averagespread")
    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(y_test, y_pred, alpha=0.5, color='teal')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Decision Tree Predictions")
    st.pyplot(fig)

    # --- Interactive Prediction ---
    st.subheader("🖊 Predict Averagespread for a New Product")
    input_data = {}
    for col in ["farmprice","atlantaretail","chicagoretail","losangelesretail","newyorkretail"]:
        input_data[col] = st.number_input(f"Enter {col}", value=float(df[col].mean()))
    product_name = st.selectbox("Select Product", df["productname"].unique())
    input_data["product_en"] = le.transform([product_name])[0]

    input_df = pd.DataFrame([input_data])
    if st.button("Predict"):
        prediction = best_dt.predict(input_df)[0]
        st.success(f"Predicted Averagespread: {prediction:.2f}")

# --- Sidebar: Visualizations ---
st.sidebar.subheader("Data Visualizations")
if st.sidebar.checkbox("Show Correlation Heatmap"):
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df_clean.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

if st.sidebar.checkbox("Show Pairplot"):
    st.subheader("Pairplot")
    st.write("Rendering large pairplots may be slow...")
    fig = sns.pairplot(df_clean)
    st.pyplot(fig)

# --- Cool Feature: Show top 10 products by average spread ---
if st.sidebar.checkbox("Top 10 Products by Average Spread"):
    st.subheader("🏆 Top 10 Products by Average Spread")
    top_products = df.groupby("productname")["averagespread"].mean().sort_values(ascending=False).head(10)
    st.bar_chart(top_products)
