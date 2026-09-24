import streamlit as st
import pickle

#loading model
with open('logistic_model.pkl','rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl','rb') as f:
    vectorizer = pickle.load(f)

#title
st.title("NLP TEXT CLASSIFICATION")

text = st.text_area("Enter Your Text")

if st.button("Predict"):

    #convert text into tf idf 

    text_tf_idf = vectorizer.transform([text])

    #prediction

    prediction = model.predict(text_tf_idf)

    st.write("Prediction: ", prediction[0])