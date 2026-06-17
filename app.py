import streamlit as st
import pickle
import pandas as pd
import numpy as np

# 1. Page Configuration for Device Responsiveness
st.set_page_config(
    page_title="Student Burnout Diagnostic Tool",
    page_icon="🎓",
    layout="wide" 
)

# 2. App Styling Headings
st.title("🎓 Smart-Campus Student Burnout Predictor")
st.markdown("This comprehensive decision-support application evaluates student behavioral and psychological metrics to assess risk levels in real time.")
st.write("---")

# 3. Load the Serialized Model Artifact
@st.cache_resource
def load_ml_pipeline():
    with open("student_burnout_model.pkl", "rb") as file_handler:
        data = pickle.load(file_handler)
    return data

try:
    pipeline = load_ml_pipeline()
    trained_model = pipeline["model"]
    model_features = pipeline["features"]
except FileNotFoundError:
    st.error(" Deployment Error: 'student_burnout_model.pkl' not detected.")
    st.stop()

# 4. Generate Interactive Input UI Fields (Divided into 3 Columns for balance)
st.subheader(" Complete Student Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("###  Behavioral & Demographics")
    age = st.slider("Student Age", min_value=18, max_value=30, value=21)
    gender = st.selectbox("Gender Component", ["Male", "Female", "Other"])
    study_hours = st.slider("Daily Study Hours", min_value=1, max_value=12, value=5)
    sleep_hours = st.slider("Daily Sleep Hours", min_value=3, max_value=10, value=7)
    screen_time = st.slider("Daily Screen Time (Hours)", min_value=1, max_value=16, value=4)
    physical_activity = st.slider("Physical Activity (Hours/Week)", min_value=0, max_value=20, value=3)

with col2:
    st.markdown("### 📝 Academic Metrics")
    cgpa = st.slider("Current Cumulative CGPA", min_value=0.0, max_value=4.0, value=3.2, step=0.01)
    attendance = st.slider("Attendance Percentage (%)", min_value=0, max_value=100, value=85)
    academic_pressure = st.slider("Academic Pressure Score", min_value=1, max_value=5, value=3)
    extracurricular = st.slider("Extracurricular Activities (Hours/Week)", min_value=0, max_value=20, value=2)
    course = st.selectbox("Academic Course Field", ["Engineering", "Computing", "Business", "Sciences", "Arts"])
    year = st.slider("Current Enrollment Year", min_value=1, max_value=5, value=2)

with col3:
    st.markdown("### 🧠 Psychological & Environmental Indicators")
    anxiety_score = st.slider("Anxiety Scale Score", min_value=1, max_value=5, value=2)
    depression_score = st.slider("Depression Scale Score", min_value=1, max_value=5, value=2)
    stress_level = st.slider("General Stress Scale Score", min_value=1, max_value=5, value=3)
    financial_stress = st.slider("Financial Stress Score", min_value=1, max_value=5, value=2)
    social_support = st.slider("Social Support Quality Score", min_value=1, max_value=5, value=4)
    sleep_quality = st.selectbox("Perceived Sleep Quality Level", ["Good", "Average", "Poor"])

# 5. Handle Categorical Values and Prediction Processing
st.write("---")
if st.button("🚀 Execute Burnout Diagnosis", use_container_width=True):
    
    # Create empty base dataframe mapping exactly to X_train shape
    input_template = pd.DataFrame(0, index=[0], columns=model_features)
    
    # Safe structural assignment helper function to prevent column errors
    def assign_safely(column_name, numeric_value):
        if column_name in input_template.columns:
            input_template[column_name] = numeric_value

    # Map the numerical entries safely
    assign_safely("age", age)
    assign_safely("cgpa", cgpa)
    assign_safely("daily_study_hours", study_hours)
    assign_safely("daily_sleep_hours", sleep_hours)
    assign_safely("attendance_percentage", attendance)
    assign_safely("screen_time_hours", screen_time)
    assign_safely("physical_activity_hours", physical_activity)
    assign_safely("anxiety_score", anxiety_score)
    assign_safely("depression_score", depression_score)
    assign_safely("academic_pressure_score", academic_pressure)
    assign_safely("financial_stress_score", financial_stress)
    assign_safely("social_support_score", social_support)
    assign_safely("extracurricular_activities_hours", extracurricular)
    assign_safely("stress_level", stress_level)
    assign_safely("year", year)
    
    # Dynamically toggle One-Hot Encoded string items safely
    for col_variant in [f"gender_{gender}", f"sleep_quality_{sleep_quality}", f"course_{course}"]:
        if col_variant in input_template.columns:
            input_template[col_variant] = 1
        
    # Process classification math via backend model pipeline
    prediction = trained_model.predict(input_template)[0]
    
    # 6. Render Output Banner
    st.subheader("🎯 Diagnostic Classification Result:")
    if prediction == 'Low':
        st.success("🟢 **Burnout Risk Level: LOW**\nThe student displays stable operational balance metrics.")
    elif prediction == 'Medium':
        st.warning("🟡 **Burnout Risk Level: MEDIUM**\nAdvise scheduling a proactive academic counseling session.")
    else:
        st.error("🔴 **Burnout Risk Level: HIGH**\nImmediate mental wellness intervention recommended.")
