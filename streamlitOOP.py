import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

class ObesityModel:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.preprocessor()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def preprocessor(self):
        """ Encoding categorical features and normalizing numerical features """
        self.categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 
                                 'SMOKE', 'SCC', 'CALC', 'MTRANS']
        self.numeric_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        
        self.encoders = {col: LabelEncoder() for col in self.categorical_cols}
        for col in self.categorical_cols:
            self.data[col] = self.encoders[col].fit_transform(self.data[col])
        
        self.scaler = StandardScaler()
        self.data[self.numeric_cols] = self.scaler.fit_transform(self.data[self.numeric_cols])
        
        self.label_encoder = LabelEncoder()
        self.data['NObeyesdad'] = self.label_encoder.fit_transform(self.data['NObeyesdad'])
    
    def train_model(self):
        """ Train the Random Forest model """
        X = self.data.drop(columns=['NObeyesdad'])
        y = self.data['NObeyesdad']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
    
    def predict(self, input_data):
        """ Predict using trained model """
        input_df = pd.DataFrame([input_data])
        for col in self.encoders:
            input_df[col] = self.encoders[col].transform([input_data[col]])
        input_df[self.numeric_cols] = self.scaler.transform(input_df[self.numeric_cols])
        pred = self.model.predict(input_df)
        prob = self.model.predict_proba(input_df)
        return self.label_encoder.inverse_transform(pred)[0], prob
    
    def show_raw_data(self):
        """ Menampilkan raw data """
        return self.data.head()

# Inisialisasi model
obesity_model = ObesityModel("https://raw.githubusercontent.com/username/repository/main/ObesityDataSet_raw_and_data_sinthetic.csv")
obesity_model.train_model()

# Streamlit App
st.title("Obesity Prediction App")
st.subheader("Raw Data")
st.dataframe(obesity_model.show_raw_data())
