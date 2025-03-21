# Streamlit App
import streamlit as st
import pandas as pd
import numpy as np
import joblib

obesity_model = joblib.load("C:\Users\krist\TugasOOP_MD.pkl")  


st.title("Obesity Prediction App")
st.subheader("Raw Data")
st.dataframe(obesity_model.show_raw_data())

