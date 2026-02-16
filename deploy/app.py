import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Add a background image using custom CSS
background_image_path = "https://images.unsplash.com/photo-1709285671893-80d2070eedf0?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
st.image(
    background_image_path, use_column_width=True
)  # Adjust 'use_column_width' based on your preference
st.title("Fraudelent Job Posting Predictor")
st.write(
    "This data product is a predictive analytics tool crafted to aid in identifying fake job postings using the best trained machine learning model."
)

# Load trained model
model = joblib.load("best_model.pkl")


def predict_fake_job_posting(job_description):
    prediction = model.predict([job_description])
    probability = model.predict_proba([job_description])[0][1]
    return prediction[0], probability


def main():
    job_description = st.text_area("Enter job description here:")
    if st.button("Predict"):
        if job_description.strip() == "":
            st.warning("Please enter a job description.")
            return

        prediction, probability = predict_fake_job_posting(job_description)
        st.subheader("Prediction Result")

        if prediction == 1:
            st.write("Prediction: ", unsafe_allow_html=True)
            st.write('<span style="color:red;">Fake</span>', unsafe_allow_html=True)
        else:
            st.write("Prediction: ", unsafe_allow_html=True)
            st.write('<span style="color:green;">Real</span>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
