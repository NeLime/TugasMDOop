import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model import DataPreprocessor, RandomForestModel

# Title and description
st.title("Machine Learning App")
st.markdown("This app will predict your **obesity level**!")

# Load dataset
df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
X = df.drop("NObeyesdad", axis=1)
y = df["NObeyesdad"]

# Preprocessing and model
preprocessor = DataPreprocessor()
X_processed = preprocessor.fit_transform(X)
model = RandomForestModel()
model.train(X_processed, y)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page:", ["Raw Data", "Data Visualization", "Obesity Prediction"])

# Raw Data Page
if page == "Raw Data":
    st.header("Raw Obesity Dataset")
    st.markdown("This is a raw data")
    st.dataframe(df)

# Data Visualization Page
elif page == "Data Visualization":
    st.header("Data Visualization")

    st.subheader("Obesity Class Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="NObeyesdad", order=df["NObeyesdad"].value_counts().index, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig)

    st.subheader("Height vs Weight by Obesity Class")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x="Height", y="Weight", hue="NObeyesdad", ax=ax2)
    st.pyplot(fig2)

# Prediction Page
elif page == "Obesity Prediction":
    st.header("Obesity Classification Prediction")
    st.write("Fill in the following information to get a prediction:")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 10, 80, 25)
        height = st.slider("Height (in meters)", 1.0, 2.5, 1.70, step=0.01)
        weight = st.slider("Weight (in kg)", 30, 200, 70)
        family_history = st.selectbox("Family history of overweight", ["yes", "no"])
        favc = st.selectbox("Frequent consumption of high calorie food", ["yes", "no"])
        fcvc = st.slider("Vegetable consumption frequency (1-3)", 1, 3, 2)

    with col2:
        ncp = st.slider("Main meals per day", 1, 4, 3)
        caec = st.selectbox("Eating between meals", ["no", "Sometimes", "Frequently", "Always"])
        smoke = st.selectbox("Smokes", ["no", "yes"])
        ch2o = st.slider("Water consumption (liters/day)", 1.0, 3.0, 2.0)
        scc = st.selectbox("Calories monitoring", ["no", "yes"])
        faf = st.slider("Physical activity frequency (times/week)", 0.0, 3.0, 1.0)
        tue = st.slider("Daily screen time (hours)", 0.0, 2.0, 1.0)
        calc = st.selectbox("Alcohol consumption", ["no", "Sometimes", "Frequently", "Always"])
        mtrans = st.selectbox("Transportation method", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

    input_data = {
        'Gender': gender,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'family_history_with_overweight': family_history,
        'FAVC': favc,
        'FCVC': fcvc,
        'NCP': ncp,
        'CAEC': caec,
        'SMOKE': smoke,
        'CH2O': ch2o,
        'SCC': scc,
        'FAF': faf,
        'TUE': tue,
        'CALC': calc,
        'MTRANS': mtrans
    }

    user_df = pd.DataFrame([input_data])

    st.subheader("Data input by user")
    st.dataframe(user_df)

    X_user_processed = preprocessor.transform(user_df)
    pred_proba = model.predict_proba(X_user_processed)[0]
    pred_class = model.predict(X_user_processed)[0]

    st.subheader("Obesity Prediction")
    proba_df = pd.DataFrame([pred_proba], columns=model.classes_)
    st.dataframe(proba_df)

    st.markdown("### The predicted output is:")
    st.success(f"{pred_class}")

