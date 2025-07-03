import streamlit as st
import pandas as pd
import joblib

# Load saved model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🎓 Prediciting Employability ")
st.write("Enter candidate information to predict employment status.")

# Sidebar info about training data
with st.sidebar:
    st.subheader("📊 Training Data Info")
    st.write("Unemployed: 140")
    st.write("Employed: 540")
    st.subheader("📊 Testing Data Info")
    st.write("Split ratio: 80% Train / 20% Test")

# Input fields
Gender = st.selectbox("Gender", ["Male", "Female"])
Age = st.selectbox("Age", ["21", "22", "23", "24", "25"])
CGPA = st.selectbox("CGPA", [
    "4.00 - 3.75", "3.74 - 3.50", "3.49 - 3.00", "2.99 - 2.50",
    "2.49 - 2.00", "1.99 - 1.50", "1.49 - 1.00", "Below 1.00"
])
Race = st.selectbox("Race", ["Malay", "Bumiputera Sabah", "Bumiputera Sarawak"])
Area_Field = st.selectbox("Area Field", [
    "SARJANA MUDA TEKNOLOGI MAKLUMAT (KEPUJIAN)",
    "SARJANA MUDA SAINS (KEPUJIAN) STATISTIK",
    "SARJANA MUDA SAINS (KEPUJIAN) SAINS AKTUARI",
    "SARJANA MUDA SAINS (KEPUJIAN) MATEMATIK PENGURUSAN",
    "SARJANA MUDA SAINS (KEPUJIAN) MATEMATIK",
    "SARJANA MUDA SAINS KOMPUTER (KEPUJIAN) PENGKOMPUTERAN NETSENTRIK",
    "SARJANA MUDA SAINS KOMPUTER (KEPUJIAN) PENGKOMPUTERAN MULTIMEDIA",
    "SARJANA MUDA SAINS KOMPUTER (KEPUJIAN) RANGKAIAN KOMPUTER",
    "SARJANA MUDA SISTEM MAKLUMAT (KEPUJIAN) PENGKOMPUTERAN PERNIAGAAN",
    "SARJANA MUDA SISTEM MAKLUMAT (KEPUJIAN) KEJURUTERAAN SISTEM MAKLUMAT",
    "SARJANA MUDA SAINS MATEMATIK PEMODELAN DAN ANALITIK (KEPUJIAN)",
    "SARJANA MUDA SAINS MAKLUMAT (KEPUJIAN) PENGURUSAN PERPUSTAKAAN",
    "SARJANA MUDA SAINS MAKLUMAT (KEPUJIAN) PENGURUSAN REKOD",
    "SARJANA MUDA SAINS MAKLUMAT (KEPUJIAN) PENGURUSAN SISTEM MAKLUMAT",
    "SARJANA MUDA SAINS MAKLUMAT (KEPUJIAN) PENGURUSAN KANDUNGAN MAKLUMAT"
])
Income = st.selectbox("Income", [
    "Kurang daripada RM1,000",
    "RM1,000 - RM1,499",
    "RM1,500 - RM1,999",
    "RM2,000 - RM2,499",
    "RM2,500 - RM2,999",
    "RM3,000 - RM3,499",
    "RM3,500 - RM3,999",
    "RM4,000 - RM4,499",
    "RM4,500 - RM4,999",
    "RM5,000 - RM5,499",
    "RM5,500 - RM5,999",
    "RM6,000 - RM6,499",
    "RM6,500 - RM6,999",
    "RM7,000 - RM7,499",
    "RM7,500 - RM7,999",
    "RM8,000 - RM8,499",
    "RM8,500 - RM8,999",
    "RM9,000 - RM9,499",
    "RM9,500 - RM9,999"
])
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
    "CGPA": manual_encode(CGPA, {
        "4.00 - 3.75": 0, "3.74 - 3.50": 1, "3.49 - 3.00": 2, "2.99 - 2.50": 3,
        "2.49 - 2.00": 4, "1.99 - 1.50": 5, "1.49 - 1.00": 6, "Below 1.00": 7
    }),
    "Race": manual_encode(Race, {"Malay": 0, "Bumiputera Sabah": 1, "Bumiputera Sarawak": 2}),
    "Area Field": manual_encode(Area_Field, {name: idx for idx, name in enumerate([
        "SARJANA MUDA TEKNOLOGI MAKLUMAT (KEPUJIAN)",
        "SARJANA MUDA SAINS (KEPUJIAN) STATISTIK",
        "SARJANA MUDA SAINS (KEPUJIAN) SAINS AKTUARI",
        "SARJANA MUDA SAINS (KEPUJIAN) MATEMATIK PENGURUSAN",
        "SARJANA MUDA SAINS (KEPUJIAN) MATEMATIK",
        "SARJANA MUDA SAINS KOMPUTER (KEPUJIAN) PENGKOMPUTERAN NETSENTRIK",
        "SARJANA MUDA SAINS KOMPUTER (KEPUJIAN) PENGKOMPUTERAN MULTIMEDIA",
        "SARJANA MUDA SAINS KOMPUTER (KEPUJIAN) RANGKAIAN KOMPUTER",
        "SARJANA MUDA SISTEM MAKLUMAT (KEPUJIAN) PENGKOMPUTERAN PERNIAGAAN",
        "SARJANA MUDA SISTEM MAKLUMAT (KEPUJIAN) KEJURUTERAAN SISTEM MAKLUMAT",
        "SARJANA MUDA SAINS MATEMATIK PEMODELAN DAN ANALITIK (KEPUJIAN)",
        "SARJANA MUDA SAINS MAKLUMAT (KEPUJIAN) PENGURUSAN PERPUSTAKAAN",
        "SARJANA MUDA SAINS MAKLUMAT (KEPUJIAN) PENGURUSAN REKOD",
        "SARJANA MUDA SAINS MAKLUMAT (KEPUJIAN) PENGURUSAN SISTEM MAKLUMAT",
        "SARJANA MUDA SAINS MAKLUMAT (KEPUJIAN) PENGURUSAN KANDUNGAN MAKLUMAT"
    ])}),
    "Income": manual_encode(Income, {name: idx for idx, name in enumerate([
        "Kurang daripada RM1,000",
        "RM1,000 - RM1,499",
        "RM1,500 - RM1,999",
        "RM2,000 - RM2,499",
        "RM2,500 - RM2,999",
        "RM3,000 - RM3,499",
        "RM3,500 - RM3,999",
        "RM4,000 - RM4,499",
        "RM4,500 - RM4,999",
        "RM5,000 - RM5,499",
        "RM5,500 - RM5,999",
        "RM6,000 - RM6,499",
        "RM6,500 - RM6,999",
        "RM7,000 - RM7,499",
        "RM7,500 - RM7,999",
        "RM8,000 - RM8,499",
        "RM8,500 - RM8,999",
        "RM9,000 - RM9,499",
        "RM9,500 - RM9,999"
    ])}),
    "Internship": manual_encode(Internship, {"Yes": 1, "No": 0}),
    "Industry Match": manual_encode(Industry_Match, {"Yes": 1, "No": 0}),
    "Skill": manual_encode(Skill, {"Low": 0, "Medium": 1, "High": 2}),
    "Extra-curricular": manual_encode(Extra_curricular, {"Yes": 1, "No": 0})
}])
# Reorder columns to match training
expected_columns = scaler.feature_names_in_
input_data = input_data[expected_columns]

# Apply scaler
input_scaled = scaler.transform(input_data)




# Prediction
if st.button("Predict Job Status"):
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled).max()

    label = "Employed" if prediction == 1 else "Unemployed"
    st.success(f"Predicted Job Status: **{label}**")
    st.info(f"Confidence: {prob:.2%}")

    st.caption("\U0001F4A1 Note: The model considers all features together. Although some attributes (like extra-curricular activities) are positive, other factors such as low skill level or lack of internship might influence the result.")
