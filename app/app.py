import streamlit as st
import pandas as pd
import joblib
import os


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="FraudShield | AI Job Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #9ca3af;
        margin-bottom: 25px;
    }

    .risk-card {
        padding: 25px;
        border-radius: 15px;
        margin: 15px 0;
    }

    .legitimate-card {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
    }

    .fraud-card {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.35);
    }

    .metric-label {
        color: #9ca3af;
        font-size: 15px;
    }

    .metric-value {
        font-size: 32px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "fake_job_detector.pkl"
)

CONFIG_PATH = os.path.join(
    BASE_DIR,
    "models",
    "model_config.pkl"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    config = joblib.load(CONFIG_PATH)

    return model, config


try:

    model, config = load_model()

    THRESHOLD = config["threshold"]

except Exception as e:

    st.error("❌ Model could not be loaded.")

    st.code(str(e))

    st.stop()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🛡️ FraudShield</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    AI-Powered Fake Job Detection System
    </div>
    """,
    unsafe_allow_html=True
)

st.info(
    "Analyze a job posting using machine learning and estimate "
    "its potential fraud risk."
)

st.divider()


# =========================================================
# JOB BASIC INFORMATION
# =========================================================

st.header("📋 Job Information")

col1, col2 = st.columns(2)


with col1:

    title = st.text_input(
        "Job Title *",
        placeholder="e.g. Data Entry Operator"
    )

    location = st.text_input(
        "Location",
        placeholder="e.g. New Delhi, India"
    )

    department = st.text_input(
        "Department",
        placeholder="e.g. Operations"
    )

    industry = st.text_input(
        "Industry",
        placeholder="e.g. Information Technology"
    )


with col2:

    function = st.text_input(
        "Job Function",
        placeholder="e.g. Administrative"
    )

    salary_range = st.text_input(
        "Salary Range",
        placeholder="e.g. 30000-50000"
    )

    has_salary = int(bool(salary_range.strip()))

    employment_type = st.selectbox(
        "Employment Type",
        [
            "",
            "Full-time",
            "Part-time",
            "Contract",
            "Temporary",
            "Other"
        ]
    )

    required_experience = st.selectbox(
        "Required Experience",
        [
            "",
            "Internship",
            "Entry level",
            "Associate",
            "Mid-Senior level",
            "Director",
            "Executive"
        ]
    )


required_education = st.selectbox(
    "Required Education",
    [
        "",
        "High School or equivalent",
        "Associate Degree",
        "Bachelor's Degree",
        "Master's Degree",
        "Doctorate",
        "Professional"
    ]
)


# =========================================================
# JOB POSTING SIGNALS
# =========================================================

st.header("⚙️ Job Posting Signals")

signal_col1, signal_col2, signal_col3 = st.columns(3)


with signal_col1:

    has_company_logo = st.checkbox(
        "🏢 Company has logo"
    )


with signal_col2:

    has_questions = st.checkbox(
        "❓ Screening questions available"
    )


with signal_col3:

    telecommuting = st.checkbox(
        "🌐 Remote / Telecommuting"
    )


# =========================================================
# TEXT INFORMATION
# =========================================================

st.header("📝 Job Description")

company_profile = st.text_area(
    "Company Profile",
    height=150,
    placeholder="Enter information about the company..."
)

description = st.text_area(
    "Job Description",
    height=220,
    placeholder="Enter the complete job description..."
)

requirements = st.text_area(
    "Requirements",
    height=180,
    placeholder="Enter required skills, qualifications and experience..."
)

benefits = st.text_area(
    "Benefits",
    height=150,
    placeholder="Enter salary, insurance, leave, perks, etc..."
)


# =========================================================
# PREPARE INPUT
# =========================================================

def prepare_input():

    title_clean = title.strip()

    company_profile_clean = company_profile.strip()

    description_clean = description.strip()

    requirements_clean = requirements.strip()

    benefits_clean = benefits.strip()


    # -----------------------------------------------------
    # Combined Text
    # -----------------------------------------------------

    combined_text = (
        title_clean + " " +
        company_profile_clean + " " +
        description_clean + " " +
        requirements_clean + " " +
        benefits_clean
    )


    # -----------------------------------------------------
    # Information Availability
    # -----------------------------------------------------

    # has_salary = int(bool(salary_range.strip()))

    has_company_profile = int(
        bool(company_profile_clean)
    )

    has_requirements = int(
        bool(requirements_clean)
    )

    has_benefits = int(
        bool(benefits_clean)
    )


    # -----------------------------------------------------
    # Missing Information
    # -----------------------------------------------------

    missing_info_count = (
        int(not bool(salary_range.strip()))
        + int(not bool(company_profile_clean))
        + int(not bool(requirements_clean))
        + int(not bool(benefits_clean))
    )


    # -----------------------------------------------------
    # Text Length Features
    # -----------------------------------------------------

    title_length = len(title_clean)

    description_length = len(description_clean)

    requirements_length = len(requirements_clean)

    benefits_length = len(benefits_clean)

    company_profile_length = len(
        company_profile_clean
    )


    # -----------------------------------------------------
    # DataFrame
    # -----------------------------------------------------

    data = {

        "title": [title_clean],

        "company_profile": [
            company_profile_clean
        ],

        "description": [
            description_clean
        ],

        "requirements": [
            requirements_clean
        ],

        "benefits": [
            benefits_clean
        ],

        "combined_text": [
            combined_text
        ],


        # Categorical
        "location": [location],

        "department": [department],

        "employment_type": [
            employment_type
        ],

        "required_experience": [
            required_experience
        ],

        "required_education": [
            required_education
        ],

        "industry": [industry],

        "function": [function],


        # Numerical
        "telecommuting": [
            int(telecommuting)
        ],

        "has_company_logo": [
            int(has_company_logo)
        ],

        "has_questions": [
            int(has_questions)
        ],

        "has_salary": [
            has_salary
        ],

        "has_company_profile": [
            has_company_profile
        ],

        "has_requirements": [
            has_requirements
        ],

        "has_benefits": [
            has_benefits
        ],

        "missing_info_count": [
            missing_info_count
        ],

        "title_length": [
            title_length
        ],

        "description_length": [
            description_length
        ],

        "requirements_length": [
            requirements_length
        ],

        "benefits_length": [
            benefits_length
        ],

        "company_profile_length": [
            company_profile_length
        ]
    }


    return pd.DataFrame(data)


# =========================================================
# ANALYZE BUTTON
# =========================================================

st.divider()

analyze = st.button(
    "🔍 Analyze Job Posting",
    type="primary",
    use_container_width=True
)


if analyze:

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    if not title.strip():

        st.warning(
            "Please enter a Job Title before analyzing."
        )

        st.stop()


    # -----------------------------------------------------
    # Prepare Input
    # -----------------------------------------------------

    input_data = prepare_input()


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    probability = model.predict_proba(
        input_data
    )[0, 1]


    prediction = int(
        probability >= THRESHOLD
    )


    # -----------------------------------------------------
    # Result
    # -----------------------------------------------------

    st.divider()

    st.header("🔎 Prediction Result")


    if prediction == 1:

        st.markdown(
            """
            <div class="risk-card fraud-card">

            <h2>🚨 POTENTIALLY FRAUDULENT JOB</h2>

            <p>
            The model detected patterns associated with
            potentially fraudulent job postings.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="risk-card legitimate-card">

            <h2>✅ LIKELY LEGITIMATE JOB</h2>

            <p>
            The model did not classify this posting as
            potentially fraudulent at the selected threshold.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    metric1, metric2, metric3 = st.columns(3)


    with metric1:

        st.metric(
            "Fraud Probability",
            f"{probability * 100:.2f}%"
        )


    with metric2:

        st.metric(
            "Classification Threshold",
            f"{THRESHOLD:.2f}"
        )


    with metric3:

        if probability < 0.25:

            risk_level = "LOW"

        elif probability < THRESHOLD:

            risk_level = "MODERATE"

        elif probability < 0.75:

            risk_level = "HIGH"

        else:

            risk_level = "VERY HIGH"


        st.metric(
            "Risk Level",
            risk_level
        )


    # -----------------------------------------------------
    # Risk Bar
    # -----------------------------------------------------

    st.subheader("📊 Fraud Risk Score")

    st.progress(
        min(float(probability), 1.0)
    )

    st.caption(
        f"Estimated fraud probability: "
        f"{probability * 100:.2f}%"
    )


    # =====================================================
    # RISK INDICATORS
    # =====================================================

    st.subheader("🚩 Risk Indicators")


    risk_items = []


    if not salary_range.strip():

        risk_items.append(
            "Salary information is missing"
        )


    if not company_profile.strip():

        risk_items.append(
            "Company profile is missing"
        )


    if not requirements.strip():

        risk_items.append(
            "Job requirements are missing"
        )


    if not benefits.strip():

        risk_items.append(
            "Benefits information is missing"
        )


    if input_data["missing_info_count"].iloc[0] >= 3:

        risk_items.append(
            "Multiple important job fields are missing"
        )


    if not has_company_logo:

        risk_items.append(
            "Company logo information is unavailable"
        )


    if risk_items:

        for item in risk_items:

            st.warning(
                f"⚠️ {item}"
            )

    else:

        st.success(
            "✅ No major missing-information indicators detected."
        )


    # =====================================================
    # POSTING SUMMARY
    # =====================================================

    st.subheader("📌 Posting Summary")


    summary_col1, summary_col2 = st.columns(2)


    with summary_col1:

        st.write(
            f"**Title:** {title}"
        )

        st.write(
            f"**Location:** {location or 'Not provided'}"
        )

        st.write(
            f"**Industry:** {industry or 'Not provided'}"
        )

        st.write(
            f"**Employment Type:** "
            f"{employment_type or 'Not provided'}"
        )


    with summary_col2:

        st.write(
            f"**Experience:** "
            f"{required_experience or 'Not provided'}"
        )

        st.write(
            f"**Education:** "
            f"{required_education or 'Not provided'}"
        )

        st.write(
            f"**Remote:** "
            f"{'Yes' if telecommuting else 'No'}"
        )

        st.write(
            f"**Salary Provided:** "
            f"{'Yes' if has_salary else 'No'}"
        )


    # =====================================================
    # MODEL INFORMATION
    # =====================================================

    with st.expander(
        "ℹ️ About FraudShield"
    ):

        st.markdown(
            f"""
            **Model:** Random Forest

            **Classification Threshold:** `{THRESHOLD:.2f}`

            The model uses:

            - Job posting text
            - TF-IDF text features
            - Categorical features
            - Engineered numerical features
            - Missing-information indicators

            The model was evaluated using a leakage-safe
            group-based train-test split.

            **Important:** This result is an ML-based risk estimate.
            It is not definitive proof that a job posting is fraudulent.
            """
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🛡️ FraudShield | Machine Learning Project | "
    "Random Forest + TF-IDF"
)