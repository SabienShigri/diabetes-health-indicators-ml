import streamlit as st
import numpy as np
import joblib

st.title("🩺 Diabetes Prediction System")

binary_model = joblib.load("models/binary_model.joblib")
multi_model = joblib.load("models/multiclass_model.joblib")
reg_model = joblib.load("models/regression_model.joblib")

tab1, tab2, tab3 = st.tabs(["Binary Classification", "Diabetes Stage", "Risk Score"])

# ---------------- BINARY ----------------
with tab1:
    st.subheader("Binary Prediction")

    age = st.slider("Age", 1, 100, 30, key="b_age")
    bmi = st.slider("BMI", 10.0, 50.0, 25.0, key="b_bmi")
    glucose = st.slider("Glucose", 50, 300, 100, key="b_glucose")

    if st.button("Predict Diabetes", key="b_btn"):
        input_data = np.array([[age, bmi, glucose]])
        pred = binary_model.predict(input_data)

        if pred[0] == 1:
            st.error("⚠️ Diabetic")
        else:
            st.success("✅ Not Diabetic")

# ---------------- MULTICLASS ----------------
with tab2:
    st.subheader("Diabetes Stage Prediction")

    age = st.slider("Age", 1, 100, 30, key="m_age")
    bmi = st.slider("BMI", 10.0, 50.0, 25.0, key="m_bmi")
    glucose = st.slider("Glucose", 50, 300, 100, key="m_glucose")

    if st.button("Predict Stage", key="m_btn"):
        input_data = np.array([[age, bmi, glucose]])
        pred = multi_model.predict(input_data)
        st.write("Predicted Stage:", pred[0])

# ---------------- REGRESSION ----------------
with tab3:
    st.subheader("Risk Score Prediction")

    age = st.slider("Age", 1, 100, 30, key="r_age")
    bmi = st.slider("BMI", 10.0, 50.0, 25.0, key="r_bmi")
    glucose = st.slider("Glucose", 50, 300, 100, key="r_glucose")

    if st.button("Predict Risk Score", key="r_btn"):
        input_data = np.array([[age, bmi, glucose]])
        pred = reg_model.predict(input_data)
        st.write("Risk Score:", float(pred[0]))