import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from model import DataPreprocessor, RandomForestModel

# Load dataset
df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

# Separate features and target
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

# Initialize preprocessor and transform the data
preprocessor = DataPreprocessor()
X_processed = preprocessor.fit_transform(X)

# Train the Random Forest model
model = RandomForestModel(n_estimators=100, random_state=42)
model.train(X_processed, y)

# Sidebar menu
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page:", ["Raw Data", "Data Visualization", "Obesity Prediction"])

# Page 1: Display raw data
if page == "Raw Data":
    st.header("📄 Raw Obesity Dataset")
    st.write("Below is the original dataset used for training:")

    st.dataframe(df)
    st.markdown(f"**Total Rows:** {df.shape[0]} &nbsp;&nbsp;&nbsp; **Total Columns:** {df.shape[1]}")

    st.subheader("Preview (Top 5 Rows)")
    st.table(df.head())

# Page 2: Data visualization
elif page == "Data Visualization":
    st.header("📊 Data Visualization")

    # Target class distribution
    st.subheader("Obesity Class Distribution")
    class_counts = df['NObeyesdad'].value_counts()
    fig1, ax1 = plt.subplots()
    sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax1)
    ax1.set_xlabel("Obesity Category")
    ax1.set_ylabel("Count")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_title("Distribution of Obesity Levels")
    st.pyplot(fig1)

    # Correlation matrix
    st.subheader("Correlation Between Numeric Features")
    numeric_cols = preprocessor.numeric_cols
    corr = df[numeric_cols].corr()
    fig2, ax2 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="Blues", ax=ax2)
    ax2.set_title("Correlation Matrix")
    st.pyplot(fig2)

    # Example: Weight distribution
    st.subheader("Weight Distribution")
    fig3, ax3 = plt.subplots()
    ax3.hist(df['Weight'], bins=20, color='skyblue', edgecolor='black')
    ax3.set_xlabel("Weight (kg)")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Weight Histogram")
    st.pyplot(fig3)

# Page 3: Prediction interface
elif page == "Obesity Prediction":
    st.header("🤖 Obesity Classification Prediction")
    st.write("Fill in the following information to get a prediction:")

    # Numeric inputs
    age = st.slider("Age", 10, 80, 25)
    height = st.slider("Height (in meters)", 1.0, 2.5, 1.70, step=0.01)
    weight = st.slider("Weight (in kg)", 30, 200, 70)
    fcvc = st.slider("Vegetable consumption frequency (1-3)", 1, 3, 2)
    ncp = st.slider("Main meals per day", 1, 4, 3)
    ch2o = st.slider("Water consumption (liters/day)", 1.0, 3.0, 2.0)
    faf = st.slider("Physical activity frequency (times/week)", 0.0, 3.0, 1.0)
    tue = st.slider("Daily screen time (hours)", 0.0, 2.0, 1.0)

    # Categorical inputs
    gender = st.selectbox("Gender", ["Male", "Female"])
    family_history = st.selectbox("Family history of overweight", ["yes", "no"])
    favc = st.selectbox("Frequent consumption of high calorie food", ["yes", "no"])
    caec = st.selectbox("Eating between meals", ["no", "Sometimes", "Frequently", "Always"])
    smoke = st.selectbox("Smokes", ["no", "yes"])
    scc = st.selectbox("Calories monitoring", ["no", "yes"])
    calc = st.selectbox("Alcohol consumption", ["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox("Transportation method", 
                           ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

    # Create input DataFrame
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

    st.subheader("Your Input:")
    st.table(user_df.T)

    # Preprocess and predict
    X_user_processed = preprocessor.transform(user_df)
    pred_proba = model.predict_proba(X_user_processed)[0]
    pred_class = model.predict(X_user_processed)[0]

    # Show prediction probabilities
    st.subheader("Prediction Probabilities:")
    prob_df = pd.DataFrame({
        'Obesity Class': model.classes_,
        'Probability (%)': (pred_proba * 100).round(2)
    })
    st.table(prob_df)

    # Final prediction
    st.subheader("Final Prediction:")
    st.markdown(f"**Predicted obesity level: `{pred_class}`**")
