import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack

# ==============================
# Page Config
st.set_page_config(page_title="JOBGUARD", layout="wide")
# ==============================

# ==============================
# Load Model + Vectorizers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "best_model.pkl"))
vectorizers = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizers.pkl"))
ohe = joblib.load(os.path.join(BASE_DIR, "ohe_encoder.pkl"))

text_cols = ["description", "title", "company_profile", "requirements", "benefits"]
cat_cols = ["employment_type", "required_experience", "country"]
bin_cols = [
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "salary_specified",
    "company_profile_specified",
]
num_cols = ["company_profile_word_count", "requirements_word_count"]

THRESHOLD = 0.4
# ==============================

# ==============================
# UI Header
# Add a background image
background_image_path = "https://images.unsplash.com/photo-1709285671893-80d2070eedf0?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
st.markdown(
    f"""
    <style>
    /* 1. Darken the entire app background */
    .stApp {{
        background: linear-gradient(rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0.75)), 
                    url("{background_image_path}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white; /* Force default text to white */
    }}

    /* 2. Style headings and labels for dark mode */
    h1, h2, h3, p, span, label {{
        color: white !important;
    }}

    /* 3. Improve contrast for input fields */
    .stTextArea textarea, .stTextInput input, [data-baseweb="select"] {{
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }}

    /* 4. Glassmorphism effect - More Transparent version */
    [data-testid="stVerticalBlock"] {{
        background-color: rgba(0, 0, 0, 0.15); /* Lowered from 0.4 to 0.15 */
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1); /* Optional: adds a faint edge */
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛡️ JOBGUARD")
st.caption("Machine Learning-Powered Fraudulent Job Posting Detection")

st.markdown("---")
# ==============================

# --- Categorical Inputs ---
country_list = [
    "us",
    "nz",
    "de",
    "gb",
    "au",
    "sg",
    "il",
    "ae",
    "ca",
    "ap",
    "eg",
    "pl",
    "gr",
    "pk",
    "mp",
    "be",
    "br",
    "sa",
    "dk",
    "ru",
    "za",
    "cy",
    "hk",
    "tr",
    "bru",
    "ie",
    "lt",
    "jp",
    "nl",
    "mh",
    "kr",
    "fr",
    "ee",
    "th",
    "pa",
    "ke",
    "mu",
    "mx",
    "ro",
    "tn",
    "fi",
    "cn",
    "in",
    "es",
    "se",
    "cl",
    "ua",
    "qa",
    "hr",
    "lv",
    "iq",
    "bg",
    "ph",
    "gj",
    "cz",
    "vi",
    "mt",
    "hu",
    "ka",
    "dl",
    "bd",
    "kw",
    "lu",
    "ng",
    "rs",
    "by",
    "vn",
    "id",
    "zm",
    "bh",
    "ug",
    "wb",
    "ch",
    "tt",
    "sd",
    "sk",
    "van",
    "ar",
    "hm",
    "kl",
    "hp",
    "tw",
    "it",
    "pt",
    "pe",
    "co",
    "si",
    "er",
    "gh",
    "no",
    "rj",
    "al",
    "at",
    "my",
    "cm",
    "sv",
    "ni",
    "lk",
    "jm",
    "kz",
    "am",
    "kh",
    "mi",
    "none",
]

# ==============================
# Input Section
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter the Job Details")
    st.caption("That you saw from a potential fraudulent job posting")
    # --- Text Inputs ---
    # title = st.text_input("Job Title")
    # description = st.text_area("Job Description")
    # requirements = st.text_area("Job Requirements")
    # company_profile = st.text_area("Company Profile")
    # benefits = st.text_area("Employee Benefits")
    # --- ROW 1 ---
    r1_col1, r1_col2 = st.columns(2)
    with r1_col1:
        title = st.text_input("Job Title", placeholder="e.g. Software Engineer")
        description = st.text_area(
            "Job Description", placeholder="Main responsibilities...", height=150
        )
    with r1_col2:
        company_name = st.text_input("Company Name", placeholder="e.g. Acme Corp")
        company_profile = st.text_area(
            "Company Profile",
            placeholder="Brief introduction of the company...",
            height=68,
        )
    # --- ROW 2 ---
    r2_col1, r2_col2 = st.columns(2)
    with r2_col1:
        requirements = st.text_area(
            "Job Requirements",
            placeholder="Skills, education, years of experience...",
            height=150,
        )
    with r2_col2:
        benefits = st.text_area(
            "Employee Benefits", placeholder="Insurance, 401k, PTO...", height=150
        )

    # --- ROW 3 ---
    r3_col1, r3_col2 = st.columns(2)
    with r3_col1:
        emp_type = st.selectbox(
            "Employment Type",
            ["Full-time", "Part-time", "Contract", "Temporary", "Other", "Unknown"],
        )
        exp = st.selectbox(
            "Experience Level",
            [
                "Internship",
                "Entry level",
                "Associate",
                "Mid-Senior level",
                "Director",
                "Executive",
                "Unknown",
            ],
        )
        country = st.selectbox(
            "Country Code",
            options=country_list,
            index=country_list.index("us") if "us" in country_list else 0,
            help="Select the 2-letter country code mentioned in the job post.",
        )
    with r3_col2:
        # --- Binary & Numerical Inputs ---
        telecommuting = st.checkbox("Telecommuting")
        has_logo = st.checkbox("Has Company Logo")
        has_questions = st.checkbox("Has Questions")
        salary_spec = st.checkbox("Salary Specified")

        profile_spec = 1 if company_profile else 0

    analyze_button = st.button("Check for Fraud")
# ==============================


# ==============================
# Prediction Logic
def transform_input(input_dict):
    # 1. Text Transformation (26,000 features)
    text_matrices = []
    for col in text_cols:
        text_matrices.append(vectorizers[col].transform([input_dict.get(col, "")]))

    # 2. Categorical Transformation (Encoded)
    cat_df = pd.DataFrame(
        [
            [
                input_dict["employment_type"],
                input_dict["required_experience"],
                input_dict["country"],
            ]
        ],
        columns=cat_cols,
    )
    X_cat = ohe.transform(cat_df.fillna("Unknown"))

    # 3. Numerical Features (Word Counts)
    desc_words = len((input_dict.get("description", "") or "").split())
    comp_words = len((input_dict.get("company_profile", "") or "").split())
    X_num = np.array([[comp_words, desc_words]])

    # 4. Binary Features
    X_bin = np.array(
        [
            [
                int(input_dict["telecommuting"]),
                int(input_dict["has_company_logo"]),
                int(input_dict["has_questions"]),
                int(input_dict["salary_specified"]),
                int(input_dict["profile_spec"]),
            ]
        ]
    )

    # 5. Final HSTACK (Must match training order exactly!)
    # X_train_desc, X_train_title, X_train_company_profile, X_train_requirements, X_train_benefits,
    # X_train_cat, X_train_num, X_train_bin
    return hstack(text_matrices + [X_cat, X_num, X_bin])


# ==============================

# ==============================
# Output Section
with col2:
    st.subheader("Detection Result")

    if analyze_button:
        # 1. Validation Check: Ensure critical fields are not empty
        if not title.strip() or not description.strip():
            st.warning(
                "⚠️ Action Required: Please enter at least a Job Title and Description to perform a fraud analysis."
            )
        else:
            # 2. Prepare the input data
            input_data = {
                "description": description,
                "title": title,
                "company_profile": company_profile,
                "requirements": requirements,
                "benefits": benefits,
                "employment_type": emp_type,
                "required_experience": exp,
                "country": country,
                "telecommuting": telecommuting,
                "has_company_logo": has_logo,
                "has_questions": has_questions,
                "salary_specified": salary_spec,
                "profile_spec": profile_spec,
            }

            # 3. Transform and Predict
            X_input = transform_input(input_data)

            # (Rest of your existing prediction logic)
            prob = model.predict_proba(X_input)[0][1]
            prediction = 1 if prob >= THRESHOLD else 0

            if prediction == 1:
                st.error(f"### 🚨 LIKELY FRAUD")
            else:
                st.success(f"### ✅ LIKELY SAFE")

            risk_score = int(prob * 100)
            if risk_score < 30:
                st.success(f"""
                    ### 🟢 LOW RISK ({risk_score}%)
                    This posting follows standard patterns for legitimate jobs. 
                    The details provided align with authentic industry listings.
                """)
            elif risk_score < 60:
                st.warning(f"""
                    ### 🟡 MODERATE RISK ({risk_score}%)
                    Some elements look unusual. This could be a poorly written 
                    ad or a potential scam. Proceed with caution and verify the source.
                """)
            else:
                st.error(f"""
                    ### 🔴 HIGH RISK ({risk_score}%)
                    **Warning:** This posting has strong characteristics of known 
                    job scams. We strongly advise against sharing personal 
                    information or financial details.
                """)

            st.markdown("---")
            st.subheader("🔍 Verify This Company")
            st.caption(
                "Ensure you apply via a verified recruiter channel. Cross-reference on trusted platforms to see official hiring posts and employee reviews:"
            )
            company_query = (
                company_name.split(" at ")[-1]
                if " at " in company_name
                else company_name
            )
            # Create dynamic search URLs
            linkedin_url = f"https://www.linkedin.com/search/results/companies/?keywords={company_query.replace(' ', '%20')}"
            indeed_url = f"https://www.indeed.com/cmp/{company_query.replace(' ', '-')}"
            glassdoor_url = f"https://www.glassdoor.com/Search/results.htm?keyword={company_query.replace(' ', '%20')}"

            indeed_logo = (
                "https://1000logos.net/wp-content/uploads/2023/01/Indeed-logo.png"
            )
            linkedin_logo = (
                "https://upload.wikimedia.org/wikipedia/commons/a/aa/LinkedIn_2021.svg"
            )
            glassdoor_logo = "https://logos-world.net/wp-content/uploads/2021/08/Glassdoor-Logo-2017.png"

            # --- Styling for Alignment and Text ---
            st.markdown(
                """
                <style>
                .verify-container {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    text-align: center;
                    padding: 10px;
                    transition: transform 0.2s;
                }
                .verify-container:hover {
                    transform: scale(1.05);
                }
                .logo-img {
                    max-height: 45px;
                    width: auto;
                    margin-bottom: 10px;
                }
                .verify-text {
                    color: #d1d1d1;
                    font-size: 0.85rem;
                    line-height: 1.2;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            # --- Layout ---
            v_col1, v_col2, v_col3 = st.columns(3)

            with v_col1:
                st.markdown(
                    f'''<a href="{linkedin_url}" target="_blank" style="text-decoration: none;">
                        <div class="verify-container" title="Check if the company has a verified business profile.">
                            <img src="{linkedin_logo}" class="logo-img">
                            <div class="verify-text">Check if the company has a verified business profile.</div>
                        </div>
                    </a>''',
                    unsafe_allow_html=True,
                )

            with v_col2:
                st.markdown(
                    f'''<a href="{indeed_url}" target="_blank" style="text-decoration: none;">
                        <div class="verify-container" title="Read employee reviews and see historical hiring data.">
                            <img src="{indeed_logo}" class="logo-img">
                            <div class="verify-text">Read employee reviews and see historical hiring data.</div>
                        </div>
                    </a>''',
                    unsafe_allow_html=True,
                )

            with v_col3:
                st.markdown(
                    f'''<a href="{glassdoor_url}" target="_blank" style="text-decoration: none;">
                        <div class="verify-container" title="Check salary ranges and anonymous company culture ratings.">
                            <img src="{glassdoor_logo}" class="logo-img">
                            <div class="verify-text">Check salary ranges and anonymous company culture ratings.</div>
                        </div>
                    </a>''',
                    unsafe_allow_html=True,
                )
# ==============================

# ==============================
# Footer
st.markdown("---")
st.caption("Model: Random Forest")
# ==============================
