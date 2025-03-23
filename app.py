import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model import DataPreprocessor, RandomForestModel

# Title 
st.title("Machine Learning App")

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
page = st.sidebar.radio("Choose a page:", ["Data Visualization", "Obesity Prediction"])


# Data Visualization
if page == "Data Visualization":
    st.info("This app will predict your obesity level!")

    with st.expander("Data"):
        st.write("This is raw data")
        st.dataframe(df)

    with st.expander("Data Visualization"):
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="Height", y="Weight", hue="NObeyesdad", palette="bright", ax=ax)
        ax.set_xlabel("Height")
        ax.set_ylabel("Weight")
        st.pyplot(fig)



# Prediction 
elif page == "Obesity Prediction":
    st.header("Obesity Classification Prediction")
    st.write("Fill in the following information to get a prediction:")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 10, 80, 25)
        height = st.slider("Height", 1.0, 2.5, 1.70, step=0.01)
        weight = st.slider("Weight", 30, 200, 70)
        family_history = st.selectbox("Family history with overweight", ["yes", "no"])
        favc = st.selectbox("FAVC", ["yes", "no"])
        fcvc = st.slider("FCVC", 1, 3, 2)

    with col2:
        ncp = st.slider("NCP", 1, 4, 3)
        caec = st.selectbox("CAEC", ["no", "Sometimes", "Frequently", "Always"])
        smoke = st.selectbox("Smoke", ["no", "yes"])
        ch2o = st.slider("CH2O", 1.0, 3.0, 2.0)
        scc = st.selectbox("SCC", ["no", "yes"])
        faf = st.slider("FAF", 0.0, 3.0, 1.0)
        tue = st.slider("TUE", 0.0, 2.0, 1.0)
        calc = st.selectbox("CALC", ["no", "Sometimes", "Frequently", "Always"])
        mtrans = st.selectbox("MTRANS", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

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

