# Streamlit App
import streamlit as st
import pickle 
import pandas as pd
import numpy as np

with open('C:\Users\krist\TugasOOP_MD.pkl', 'rb') as f:
    obesity_model = pickle.load(f)

st.title("Obesity Prediction App")
st.subheader("Raw Data")
st.dataframe(obesity_model.show_raw_data())