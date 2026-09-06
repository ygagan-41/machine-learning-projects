import pandas as pd
import streamlit as st
import joblib

#load saved model , scaler and expected columns
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

#title denge
st.title("HEART PREDICTION SYSTEM")
st.markdown("provide your following details to check your heart stroke risk")

#now we will collect input details

#use slider for age
age = st.slider('Age',18,100,40)

#now we will use selectbox
sex = st.selectbox("Sex",["MALE","FEMALE"])
chest_pain = st.selectbox("Chest Pain Type",["ATA","NAP","TA","ASY"])

#now we will use input 
resting_bp = st.number_input("Resting Blood Pressure(mm Hg)",80,200,120)
cholesterol = st.number_input("Cholesterol (mg/dL)",100,600,200)

fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL",[0,1])
resting_ecg = st.selectbox("Resting ECG",["Normal","ST","LVH"])

max_hr = st.slider("Max Heart Rate",60,220,150)
exercise_angina = st.selectbox('Exercise-Induced Angina',["Y","N"])

oldpeak = st.slider("Oldpeak (ST Depression)",0.0,6.0,1.0)
st_slope = st.selectbox("ST slope",["UP","FLAT","DOWN"])

#when predict is clicked
if st.button("Predict"):

    #we will create dictionary to give input
    raw_input = {
        'Age':age,
        'RestingBp':resting_bp,
        'Cholesterol':cholesterol,
        'MaxHR':max_hr,
        'Oldpeak':oldpeak,
        'Sex_'+ sex:1,
        'ChestPainType_'+ chest_pain:1,
        'RestingECG_'+ resting_ecg:1,
        'ExerciseAngina_'+ exercise_angina:1,
        'ST_Slope_'+ st_slope:1
    }

    #now we will create raw input into dataframe
    input_df = pd.DataFrame([raw_input])

    #Fill in missing columns with 0s
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    #reorder columns
    input_df = input_df[expected_columns]

    #now we will scale the input
    scaled_input = scaler.transform(input_df)

    #now we will make prediction 
    #is line ka matlab hai ki jab model predict 
    #karega, prediction array se first value ko loega or prediction me stored kardega 
    prediction = model.predict(scaled_input)[0]

    #now we will show the result
    if prediction == 1:
        st.error("⚠️HIGH RISK OF HEART DISEASE")
    else:
        st.success("✅LOW RISK OF HEART DISEASE")