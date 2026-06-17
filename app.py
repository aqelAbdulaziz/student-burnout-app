# =====================================================================
# DEPLOYMENT STEP 2: STREAMLIT WEB APP ENGINE (MOBILE-RESPONSIVE FRONTEND)
# =====================================================================
import streamlit as pd
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# 1. Page Configuration for Device Responsiveness
st.set_page_config(
    page_title="Student Burnout Diagnostic Tool",
    page_icon="🎓",
    layout="centered" # Keeps UI clustered perfectly on mobile screens
)

# 2. App Styling Headings
st.title("🎓 Smart-Campus Student Burnout Predictor")
st.markdown("This decision-support application evaluates student behavioral metrics to assess risk levels in real time.")
st.write("---")

# 3. Securely Load the Serialized Model Artifact
@st.cache_resource # Prevents reloading the file on every click
def load_ml_pipeline():
    with open("student_burnout_model.pkl", "rb") as file_handler:
        data = pickle.load(file_handler)
    return data

try:
    pipeline = load_ml_pipeline()
    trained_model = pipeline["model"]
    model_features = pipeline["features"]
except FileNotFoundError:
    st.error("🚨 Deployment Error: 'student_burnout_model.pkl' not detected in working directory.")
    st.stop()

# 4. Generate Interactive Input UI Fields
st.subheader("📊 Enter Student Parameters")

# Organize inputs into two columns for layout balance
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Student Age", min_value=18, max_value=30, value=21)
    cgpa = st.slider("Current Cumulative CGPA", min_value=0.0, max_value=4.0, value=3.0, step=0.01)
    study_hours = st.slider("Daily Study Hours", min_value=1, max_value=12, value=4)
    sleep_hours = st.slider("Daily Sleep Hours", min_value=3, max_value=10, value=7)

with col2:
    attendance = st.slider("Attendance Percentage (%)", min_value=0, max_value=100, value=85)
    screen_time = st.slider("Daily Screen Time (Hours)", min_value=1, max_value=16, value=5)
    physical_activity = st.slider("Physical Activity (Hours/Week)", min_value=0, max_value=20, value=3)

# 5. Handle Categorical Values safely using a Form submit button
st.write("---")
if st.button("🚀 Execute Burnout Diagnosis"):
    
    # Re-create an empty dataframe row matching the exact structure used during model training
    input_template = pd.DataFrame(0, index=[0], columns=model_features)
    
    # Fill in the continuous numerical features
    input_template["age"] = age
    input_template["cgpa"] = cgpa
    input_template["daily_study_hours"] = study_hours
    input_template["daily_sleep_hours"] = sleep_hours
    input_template["attendance_percentage"] = attendance
    input_template["screen_time_hours"] = screen_time
    input_template["physical_activity_hours"] = physical_activity
    
    # Run the math equation through the saved model
    prediction = trained_model.predict(input_template)[0]
    
    # 6. Render Output Banner cleanly based on categorical prediction result
    st.subheader("🎯 Diagnostic Classification Result:")
    if prediction == 'Low':
        st.success("🟢 **Burnout Risk Level: LOW**\nThe student displays stable operational balance metrics.")
    elif prediction == 'Medium':
        st.warning("🟡 **Burnout Risk Level: MEDIUM**\nAdvise scheduling a proactive academic counseling session.")
    else:
        st.error("🔴 **Burnout Risk Level: HIGH**\nImmediate mental wellness intervention recommended.")