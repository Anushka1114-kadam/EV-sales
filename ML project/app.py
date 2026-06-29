import streamlit as st
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Title
# ----------------------------
st.title("🚗 EV Sales Prediction App")
st.write("Enter vehicle details to predict EV Sales")
st.sidebar.header("Dashboard Menu")

page = st.sidebar.radio(
    "Select",
    ["Navigation", "Dashboard", "Prediction"]
)
# -----------------------------
# DASHBOARD
# ---------------------------
# --

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_excel("ev_sales_dataset.xlsx")
if page == "Dashboard":

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

col1, col2, col3 = st.columns(3)
col1.metric("Total Records",len(df))

if "Sales" in df.columns:
        col2.metric("Total Sales", int(df["Sales"].sum()))

if "Brand" in df.columns:
        col3.metric("Brands", df["Brand"].nunique())

st.markdown("---")

if "Sales" in df.columns and "year" in df.columns:
        st.subheader("Sales Trend")

        fig = px.line(
            df,
            x="Year",
            y="Sales",
            markers=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

if "Brand" in df.columns and "Sales" in df.columns:

        st.subheader("Sales by Brand")

        brand_sales = df.groupby("Brand")["Sales"].sum().reset_index()

        fig = px.bar(
            brand_sales,
            x="Brand",
            y="Sales",
            color="Brand"
        )

        st.plotly_chart(fig, use_container_width=True)

if "Price_USD" in df.columns and "Sales" in df.columns:

        st.subheader("Price vs Sales")

        fig = px.scatter(
            df,
            x="Price_USD",
            y="Sales",
            color="Brand",
            size="Battery_Capacity_kWh"
        )

        st.plotly_chart(fig, use_container_width=True)

if "Battery_Capacity_kWh" in df.columns:

        st.subheader("Battery Capacity Distribution")

        fig = px.histogram(
            df,
            x="Battery_Capacity_kWh",
            nbins=20
        )
st.plotly_chart(fig, use_container_width=True)

st.subheader("Correlation Heatmap")

numeric = df.select_dtypes(include="number")
fig, ax = plt.subplots(figsize=(12,8))

sns.heatmap(
        numeric.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

st.pyplot(fig)
        

# ----------------------------
# Encode Categorical Columns
# ----------------------------
brand_encoder = LabelEncoder()
model_encoder = LabelEncoder()

df["Brand"] = brand_encoder.fit_transform(df["Brand"])
df["Model"] = model_encoder.fit_transform(df["Model"])

# ----------------------------
# Features and Target
# ----------------------------
X = df[[
    "Year",
    "Brand",
    "Model",
    "Battery_Capacity_kWh",
    "Range_km",
    "Charging_Time_hr",
    "Price_USD",
    "Horsepower",
    "Num_Charging_Stations",
    "Govt_Subsidy_USD",
    "Fuel_Price_Index",
    "Marketing_Spend_USD",
    "Competitor_Count",
    "Avg_Income_Region_USD",
    "Unemployment_Rate"
]]

y = df["Sales"]

# ----------------------------
# Train Model
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------
# User Input
# ----------------------------

year = st.number_input("Year",2020,2035,2025)

brand = st.selectbox(
    "Brand",
    brand_encoder.classes_
)

model_name = st.selectbox(
    "Model",
    model_encoder.classes_
)

battery = st.number_input("Battery Capacity (kWh)")

range_km = st.number_input("Range (km)")

charging = st.number_input("Charging Time (Hours)")

price = st.number_input("Price (USD)")

horsepower = st.number_input("Horsepower")

stations = st.number_input("Charging Stations")

subsidy = st.number_input("Government Subsidy (USD)")

fuel = st.number_input("Fuel Price Index")

marketing = st.number_input("Marketing Spend (USD)")

competitor = st.number_input("Competitor Count")

income = st.number_input("Average Income")

unemployment = st.number_input("Unemployment Rate")

# Encode

brand = brand_encoder.transform([brand])[0]
model_code = model_encoder.transform([model_name])[0]

# ----------------------------
# Prediction
# ----------------------------

if st.button("Predict Sales"):

    input_df = pd.DataFrame([[
        year,
        brand,
        model_code,
        battery,
        range_km,
        charging,
        price,
        horsepower,
        stations,
        subsidy,
        fuel,
        marketing,
        competitor,
        income,
        unemployment
    ]], columns=X.columns)

    prediction = model.predict(input_df)[0]

    st.success(f"Predicted EV Sales = {prediction:.2f}")
    

