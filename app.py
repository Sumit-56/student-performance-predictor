import streamlit as st
import numpy as np
import joblib

# ==========================================
# 1. LOAD THE TRAINED MODEL & SCALER
# ==========================================
model = joblib.load('student_rf_model.pkl') # Ensure this name matches!
scaler = joblib.load('student_scaler_FINAL.pkl') # The new 6-feature scaler!

# ==========================================
# 2. DESIGN THE WEB APP GUI
# ==========================================
st.set_page_config(page_title="Student Grade Predictor", page_icon="🎓", layout="wide")
st.title("🎓 Student Performance Predictor")
st.write("Enter the student's data below to predict their final letter grade.")

st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Demographics")
    age = st.number_input("Student Age", min_value=10, max_value=30, value=18)
    
    # EXACT mappings from your screenshot
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    gender_enc = {"Female": 0, "Male": 1, "Other": 2}[gender]

    school_type = st.selectbox("School Type", ["Private", "Public"])
    school_enc = {"Private": 0, "Public": 1}[school_type]
    
    parent_edu = st.selectbox("Parent Education Level", ["Diploma", "Graduate", "High School", "No Formal", "PhD", "Post Graduate"])
    parent_enc = {"Diploma": 0, "Graduate": 1, "High School": 2, "No Formal": 3, "PhD": 4, "Post Graduate": 5}[parent_edu]

with col2:
    st.subheader("📚 Academic Habits")
    study_hours = st.slider("Weekly Study Hours", 0.0, 50.0, 15.0)
    attendance = st.slider("Attendance Percentage (%)", 0.0, 100.0, 85.0)
    
    study_method = st.selectbox("Primary Study Method", ["Coaching", "Group Study", "Mixed", "Notes", "Online Videos", "Textbook"])
    method_enc = {"Coaching": 0, "Group Study": 1, "Mixed": 2, "Notes": 3, "Online Videos": 4, "Textbook": 5}[study_method]

    travel_time = st.selectbox("Travel Time to School", ["15-30 min", "30-60 min", "<15 min", ">60 min"])
    travel_enc = {"15-30 min": 0, "30-60 min": 1, "<15 min": 2, ">60 min": 3}[travel_time]

with col3:
    st.subheader("🌍 External Factors & Scores")
    internet = st.selectbox("Internet Access at Home", ["No", "Yes"])
    internet_enc = {"No": 0, "Yes": 1}[internet]
    
    extra_act = st.selectbox("Participates in Extracurriculars?", ["No", "Yes"])
    extra_enc = {"No": 0, "Yes": 1}[extra_act]

    math_score = st.number_input("Math Score", min_value=0.0, max_value=100.0, value=75.0)
    science_score = st.number_input("Science Score", min_value=0.0, max_value=100.0, value=75.0)
    english_score = st.number_input("English Score", min_value=0.0, max_value=100.0, value=75.0)

st.markdown("---")

# ==========================================
# 3. PREDICTION LOGIC
# ==========================================
if st.button("Predict Final Grade", type="primary"):
    
    # 1. Isolate ONLY the 3 features the scaler knows about!
    features_to_scale = np.array([[age, study_hours, attendance]])
    
    # Scale them
    scaled_features = scaler.transform(features_to_scale)[0]
    scaled_age = scaled_features[0]
    scaled_study_hours = scaled_features[1]
    scaled_attendance = scaled_features[2]
    
    # 2. Reconstruct the full 13-feature array in the exact order the model expects
    user_data = np.array([[
        scaled_age,         # 1. age (scaled)
        gender_enc,         # 2. gender
        school_enc,         # 3. school_type
        parent_enc,         # 4. parent_education
        scaled_study_hours, # 5. study_hours (scaled)
        scaled_attendance,  # 6. attendance_percentage (scaled)
        internet_enc,       # 7. internet_access
        travel_enc,         # 8. travel_time
        extra_enc,          # 9. extra_activities
        method_enc,         # 10. study_method
        math_score,         # 11. math_score (unscaled)
        science_score,      # 12. science_score (unscaled)
        english_score       # 13. english_score (unscaled)
    ]])
    
    # Make the prediction
    prediction = model.predict(user_data)
    
    # Translate the predicted number back to the correct letter grade
    grade_mapping = {
        0: "A (Excellent)",
        1: "B (Good)",
        2: "C (Average)",
        3: "D (Below Average)",
        4: "E (Poor)",
        5: "F (Fail)"
    }
    
    final_grade = grade_mapping.get(prediction[0], "Unknown Grade")
    
    st.success(f"🤖 The Artificial Intelligence predicts this student will receive a Grade Category: **{final_grade}**")
    st.balloons()