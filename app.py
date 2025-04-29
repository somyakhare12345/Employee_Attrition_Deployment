
import time
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from io import BytesIO
import matplotlib.pyplot as plt

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(page_title="🌟 Employee Attrition App", layout="wide")

# ----------------------------
# Session States
# ----------------------------
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if 'logout' not in st.session_state:
    st.session_state.logout = False

if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0

if 'last_activity' not in st.session_state:
    st.session_state.last_activity = time.time()

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# ----------------------------
# Session Timeout
# ----------------------------
session_timeout = 30 * 60  # 30 minutes
if time.time() - st.session_state.last_activity > session_timeout:
    st.session_state.authenticated = False
    st.session_state.logout = True
    st.rerun()

# ----------------------------
# Authentication
# ----------------------------
def login_page():
    st.title("🔐 Employee Attrition Prediction Login")
    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")
    if st.button("🔓 Login"):
        if username == "admin" and password == "password":
            st.session_state.authenticated = True
            st.success("✅ Login successful!")
            st.session_state.last_activity = time.time()
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

def logout():
    st.session_state.authenticated = False
    st.session_state.logout = True

# ----------------------------
# Dark Mode Styling
# ----------------------------
def set_dark_mode():
    if st.session_state.dark_mode:
        st.markdown("""
        <style>
        html, body, [class*="css"]  {
            background-color: #181818 !important;
            color: #ffffff !important;
        }
        .stButton > button {
            background-color: #333333;
            color: white;
        }
        .stTextInput input, .stSelectbox div, .stSlider > div {
            background-color: #2e2e2e;
            color: white;
        }
        .stDownloadButton > button {
            background-color: #444;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

# ----------------------------
# Prediction Page
# ----------------------------
def prediction_app():
    st.title("📊 Employee Attrition Prediction")

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("model_columns.pkl", "rb") as f:
        model_columns = pickle.load(f)

    st.subheader("📝 Input Employee Information")

    with st.expander("🔧 Customize Input Features", expanded=True):
        def user_input_features():
            with st.container():
                col1, col2 = st.columns(2)

                with col1:
                    Age = st.slider("📅 Age", 18, 60, 30)
                    DistanceFromHome = st.slider("📍 Distance From Home", 1, 30, 5)
                    Education = st.selectbox("🎓 Education", ["Below College", "College", "Bachelor", "Master", "Doctor"])
                    EnvironmentSatisfaction = st.selectbox("🌿 Environment Satisfaction", ["Low", "Medium", "High", "Very High"])
                    JobInvolvement = st.selectbox("💼 Job Involvement", ["Low", "Medium", "High", "Very High"])
                    JobLevel = st.selectbox("📊 Job Level", ["Junior Level", "Mid Level", "Senior Level", "Executive Level"])
                    JobRole = st.selectbox("🧑‍🔬 Job Role", [
                        "Human Resources", "Laboratory Technician", "Manager", "Manufacturing Director",
                        "Research Director", "Research Scientist", "Sales Executive", "Sales Representative"
                    ])

                with col2:
                    MaritalStatus = st.selectbox("💍 Marital Status", ["Single", "Married", "Divorced"])
                    MonthlyIncome = st.number_input("💵 Monthly Income", 1000, 20000, 5000)
                    NumCompaniesWorked = st.slider("🏢 Number of Companies Worked", 0, 9, 2)
                    OverTime = st.selectbox("⏳ OverTime", ["Yes", "No"])
                    TotalWorkingYears = st.slider("🧮 Total Working Years", 0, 40, 10)
                    TrainingTimesLastYear = st.slider("📚 Training Times Last Year", 0, 6, 2)
                    WorkLifeBalance = st.selectbox("⚖️ Work Life Balance", ["Bad", "Good", "Better", "Best"])

            data = {
                "Age": Age,
                "DistanceFromHome": DistanceFromHome,
                "Education": Education,
                "EnvironmentSatisfaction": EnvironmentSatisfaction,
                "JobInvolvement": JobInvolvement,
                "JobLevel": JobLevel,
                "JobRole": JobRole,
                "MaritalStatus": MaritalStatus,
                "MonthlyIncome": MonthlyIncome,
                "NumCompaniesWorked": NumCompaniesWorked,
                "OverTime": OverTime,
                "TotalWorkingYears": TotalWorkingYears,
                "TrainingTimesLastYear": TrainingTimesLastYear,
                "WorkLifeBalance": WorkLifeBalance
            }
            return pd.DataFrame([data])

        input_df = user_input_features()
        input_data_encoded = pd.get_dummies(input_df)
        input_data_encoded = input_data_encoded.reindex(columns=model_columns, fill_value=0)
        input_scaled = scaler.transform(input_data_encoded)
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

    st.subheader("🔍 Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Employee likely to **LEAVE**.")
    else:
        st.success("✅ Employee likely to **STAY**.")

    st.info(f"📈 Staying Probability: `{prediction_proba[0][0]*100:.2f}%`")
    st.info(f"📉 Leaving Probability: `{prediction_proba[0][1]*100:.2f}%`")

    result_df = input_df.copy()
    result_df['Attrition Prediction'] = ['Yes' if prediction[0] == 1 else 'No']
    result_df['Attrition Probability'] = [round(prediction_proba[0][1], 2)]

    csv = result_df.to_csv(index=False).encode()
    st.download_button("📥 Download Report", data=csv, file_name="prediction_report.csv", mime="text/csv")

    st.session_state.total_predictions += 1

# ----------------------------
# Model Comparison Page
# ----------------------------
def model_comparison():
    st.title("📈 Model Comparison & Evaluation")

    st.subheader("📊 Accuracy Scores of ML Models")

    data = {
        'Algorithms': [
            'Logistic Regression',
            'Random Forest',
            'Support Vector Machine',
            'XGBoost',
            'LightGBM',
            'CatBoost',
            'AdaBoost'
        ],
        'Training Data Accuracy Score': [
            0.8980,
            0.8688,
            0.8970,
            1.0000,
            1.0000,
            0.9689,
            0.8989
        ],
        'Testing Data Accuracy Score': [
            0.8571,
            0.8390,
            0.8662,
            0.8186,
            0.8367,
            0.8458,
            0.8209
        ]
    }

    df_results = pd.DataFrame(data)
    st.dataframe(df_results, use_container_width=True)

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("🧭 Navigation")

if st.session_state.authenticated:
    st.sidebar.markdown("### 👋 Welcome, Admin!")
    st.sidebar.markdown(f"### 📊 Total Predictions Made: `{st.session_state.total_predictions}`")

    st.sidebar.subheader("🌓 Dark Mode")
    st.session_state.dark_mode = st.sidebar.checkbox("Enable Dark Mode")
    set_dark_mode()

    st.sidebar.subheader("🛠️ Help")
    st.sidebar.markdown("For more info, visit the [📄 documentation](https://example.com).")

    st.sidebar.subheader("⭐ Rate Your Experience")
    rating = st.sidebar.slider("Rate the app", 1, 5)
    if rating:
        st.sidebar.markdown(f"🙏 Thanks for rating us {rating}/5!")

    page = st.sidebar.radio("🔀 Go to", ["Prediction", "Model Comparison"])
    st.sidebar.button("🚪 Logout", on_click=logout)

    if page == "Prediction":
        prediction_app()
    elif page == "Model Comparison":
        model_comparison()
else:
    login_page()

if st.session_state.logout:
    st.session_state.logout = False
    st.rerun()

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("<center>Made with ❤️ by Somya Khare</center>", unsafe_allow_html=True)
