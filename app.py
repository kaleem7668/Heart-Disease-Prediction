import streamlit as st
import pandas as pd
import  joblib

# Load saved model, scaler, and expected columns
model = joblib.load('Logistic_heart.pkl')
scaler = joblib.load('scaler.pkl')
excepted_columns = joblib.load('columns.pkl')


st.title('Heart Predictions by MKKHMG')
st.write('Provide the following details to check your heart strok risk:')

# Collect user input
age = st.number_input('Age', 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chess_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input('Resting Blood Pressure (mm Hg)', 80, 200, 120)
cholestrol = st.number_input('Cholesterol (mg/dl)', 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["0", "1"])
reasting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider('Max Heart Rate', 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
oldpeak = st.slider('Oldpeak (ST Depression)', 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# when the 'Predict' button is clicked
if st.button('Predict'):
  # create a raw input dictionary
  raw_input = {
    'Age': age,
    "resting_bp" : resting_bp,
    "cholestrol" : cholestrol,
    "fasting_bs": fasting_bs,
    "max_hr": max_hr,
    "oldpeak": oldpeak,
    "Sex_" + sex: 1,
    "ChestPainType_" + chess_pain: 1,
    "RestingECG_" + reasting_ecg: 1,
    "ExerciseAngina_" + exercise_angina: 1,
    "ST_Slope_" + st_slope: 1
  }

# create a input  dataframe 
input_df = pd.DataFrame([raw_input])

# Fill in missing columns with 0s
for col in excepted_columns:
  if col not in input_df.columns:
    input_df[col] = 0

  # Reorder columns 
input_df = input_df[excepted_columns]

# Scale the input data
scale_input = scaler.transform(input_df)

  # Make prediction
prediction = model.predict(scale_input)[0]

# show the prediction result
if prediction == 1:
    st.error('You have a high risk of heart disease. Please consult a doctor.')
else:
    st.success('You have a low risk of heart disease. Keep maintaining a healthy lifestyle!')



