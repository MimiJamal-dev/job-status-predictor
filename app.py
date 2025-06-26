import streamlit as st
import pandas as pd
import joblib

# Load saved model
model = joblib.load("best_model.pkl")

st.title("🎓 Job Status Predictor")
st.write("Enter candidate information to predict employment status.")

# Input fields
Gender = st.selectbox("Gender", ["Male", "Female"])
Age = st.selectbox("Age", ["21", "22", "23", "24", "25"])  # Adjust based on real data
CGPA = st.selectbox("CGPA", ["2.5-2.99", "3.0-3.49", "3.5-4.0"])
Race = st.selectbox("Race", ["Malay", "Chinese", "Indian", "Others"])
Area_Field = st.selectbox("Area Field", ["IT", "Engineering", "Business", "Education"])
Income = st.selectbox("Income", ["<1k", "1k-2k", "2k-3k", ">3k"])
Internship = st.selectbox("Internship Experience", ["Yes", "No"])
Industry_Match = st.selectbox("Industry matches field of study?", ["Yes", "No"])
Skill = st.selectbox("Skill Level", ["Low", "Medium", "High"])
Extra_curricular = st.selectbox("Involved in Extra-curricular?", ["Yes", "No"])

# Label encoding simulation (must match training encoding!)
def manual_encode(val, mapping):
    return mapping.get(val, 0)

input_data = pd.DataFrame([{
    "Gender": manual_encode(Gender, {"Male": 1, "Female": 0}),
    "Age": manual_encode(Age, {"21": 0, "22": 1, "23": 2, "24": 3, "25": 4}),
    "CGPA": manual_encode(CGPA, {"2.5-2.99": 0, "3.0-3.49": 1, "3.5-4.0": 2}),
    "Race": manual_encode(Race, {"Malay": 0, "Chinese": 1, "Indian": 2, "Others": 3}),
    "Area Field": manual_encode(Area_Field, {"IT": 0, "Engineering": 1, "Business": 2, "Education": 3}),
    "Income": manual_encode(Income, {"<1k": 0, "1k-2k": 1, "2k-3k": 2, ">3k": 3}),
    "Internship": manual_encode(Internship, {"Yes": 1, "No": 0}),
    "Industry Match": manual_encode(Industry_Match, {"Yes": 1, "No": 0}),
    "Skill": manual_encode(Skill, {"Low": 0, "Medium": 1, "High": 2}),
    "Extra-curricular": manual_encode(Extra_curricular, {"Yes": 1, "No": 0})
}])

# Prediction
if st.button("Predict Job Status"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data).max()

    label = "Employed" if prediction == 1 else "Unemployed"
    st.success(f"Predicted Job Status: **{label}**")
    st.info(f"Confidence: {prob:.2%}")
