"""
app.py
======
AI Code Safety Inspector Dashboard
Matches exact reference structure with:
- Single-box Code Input & File Upload
- ML Defect Probability Scoring (STATUS: SAFE TO SHIP / NEEDS REVIEW)
- Explanation & Fix Code with:
  • Key Enhancements
  • 🛠️ Refactored Code
  • 🔍 Root-Cause Diagnosis
  • 🧪 Automated Unit Tests

Run with:  streamlit run app.py
"""

import os
import sys
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ai_assistant import get_ai_explanation_and_fix
from src.analyzer import analyze_code_lines
from src.features import DEVELOPER_FEATURES
from src.predict import BugPredictor

MODELS_DIR = "models"
TEST_CSV = "data/flask_commits_test.csv"

st.set_page_config(
    page_title="AI Code Safety Inspector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Styling to match the clean dark UI
st.markdown("""
<style>
    /* Global styles */
    .block-container {
        max-width: 850px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .header-title {
        text-align: center;
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .header-subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    /* Primary button styling */
    div.stButton > button:first-child {
        background-color: #ff4b4b;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.05rem;
        padding: 0.6rem 1.2rem;
        width: 100%;
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #ff3333;
        color: white;
        transform: translateY(-1px);
    }
    /* Status Cards */
    .status-card-safe {
        background-color: #0b291d;
        border: 1px solid #10b981;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    .status-card-risk {
        background-color: #381212;
        border: 1px solid #ef4444;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    .status-title-safe {
        color: #34d399;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .status-title-risk {
        color: #f87171;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .status-detail-safe {
        color: #6ee7b7;
        font-size: 0.95rem;
        margin-bottom: 0.4rem;
    }
    .status-detail-risk {
        color: #fca5a5;
        font-size: 0.95rem;
        margin-bottom: 0.4rem;
    }
    .status-action {
        color: #d1d5db;
        font-size: 0.95rem;
    }
    /* Sub-section headers */
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-subtitle {
        color: #9ca3af;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }
    .enhancement-list {
        margin-bottom: 1.2rem;
        line-height: 1.7;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    return BugPredictor(MODELS_DIR)


@st.cache_data
def load_test_commits():
    if not os.path.exists(TEST_CSV):
        return None
    df = pd.read_csv(TEST_CSV)
    df["added_code"] = df["added_code"].fillna("")
    df["removed_code"] = df["removed_code"].fillna("")
    return df


def main():
    # Header Section
    st.markdown('<div class="header-title">🛡️ AI Code Safety Inspector</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Detect potential bugs, analyze risk likelihood, and generate automated explanations with fixed code.</div>', unsafe_allow_html=True)

    try:
        predictor = load_predictor()
    except Exception as e:
        st.error(f"Could not load AI models: {e}")
        st.stop()

    # Session State
    if "code_input" not in st.session_state:
        st.session_state["code_input"] = 'a = 5\nb = 3\nsum = a + b\nprint("Sum:", sum)'

    tab_check, tab_history = st.tabs(["📝 Check Your Code / File", "📜 Browse Real Historical Samples"])

    with tab_check:
        # File Upload Box
        st.markdown("📂 **Upload or drag & drop a code file (.py, .txt):**")
        uploaded_file = st.file_uploader(
            "Upload",
            type=["py", "txt", "js", "java"],
            help="Supported formats: Python, JavaScript, Java, Text",
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            try:
                st.session_state["code_input"] = uploaded_file.read().decode("utf-8")
                st.success(f"Loaded `{uploaded_file.name}` successfully!")
            except Exception as e:
                st.error(f"Error reading file: {e}")

        # Code Input Area
        st.markdown("Paste your code snippet here:")
        code_text = st.text_area(
            "Paste your code snippet here:",
            value=st.session_state["code_input"],
            height=160,
            label_visibility="collapsed",
            placeholder='a = 5\nb = 3\nsum = a + b\nprint("Sum:", sum)'
        )

        # Primary Action Button
        check_clicked = st.button("🔍 Check Code Safety", type="primary", use_container_width=True)

        if check_clicked:
            if not code_text.strip():
                st.warning("Please enter or upload code first.")
            else:
                # 1. Run Local Deep Learning Model Prediction
                dev_values = dict(predictor.default_dev_features())
                ml_res = predictor.predict(added_code=code_text, removed_code="", dev_features=dev_values)
                
                # 2. Run Line Analyzer
                line_issues = analyze_code_lines(code_text)
                has_high_issues = any(i["severity"] == "High" for i in line_issues)
                
                # Compute risk percentage
                risk_prob = ml_res["hybrid_probability"] * 100
                if not has_high_issues and risk_prob < 15.0:
                    risk_prob = 5.0
                    is_clean = True
                elif has_high_issues:
                    risk_prob = max(risk_prob, 76.0)
                    is_clean = False
                else:
                    is_clean = (risk_prob < 50.0)

                st.markdown('<div class="section-title">📋 Safety Assessment Report</div>', unsafe_allow_html=True)

                # Render Status Card matching Image 1
                if is_clean:
                    st.markdown(f"""
                    <div class="status-card-safe">
                        <div class="status-title-safe">🟢 STATUS: SAFE TO SHIP</div>
                        <div class="status-detail-safe"><b>Defect Likelihood:</b> {risk_prob:.0f}% (Low Risk)</div>
                        <div class="status-action"><b>Action:</b> ✅ Clean structure. Standard automated testing is sufficient.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="status-card-risk">
                        <div class="status-title-risk">🔴 STATUS: NEEDS REVIEW</div>
                        <div class="status-detail-risk"><b>Defect Likelihood:</b> {risk_prob:.0f}% (High Risk)</div>
                        <div class="status-action"><b>Action:</b> 🚨 High defect likelihood or anti-patterns found. Peer review & test coverage required.</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")

                # 3. Fetch AI Explanation & Fix
                with st.spinner("Generating architectural explanation & production-hardened fix..."):
                    ai_data = get_ai_explanation_and_fix(code_text, is_clean=is_clean, risk_prob=risk_prob)

                # Explanation & Fix Code Section matching Image 2, 3, 4
                st.markdown('<div class="section-title">💡 Explanation & Fix Code</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-subtitle">Detailed architectural explanation, production-hardened replacement code, and automated unit tests.</div>', unsafe_allow_html=True)

                # Key Enhancements
                st.markdown("#### ⚡ Key Enhancements:")
                enhancements = ai_data.get("key_enhancements", [])
                for enh in enhancements:
                    st.markdown(f"• {enh}")

                st.write("")

                # Sub-tabs for Refactored Code, Root-Cause Diagnosis, Automated Unit Tests
                subtab1, subtab2, subtab3 = st.tabs([
                    "🛠️ Refactored Code",
                    "🔍 Root-Cause Diagnosis",
                    "🧪 Automated Unit Tests"
                ])

                with subtab1:
                    st.code(ai_data.get("refactored_code", "# No code generated"), language="python")

                with subtab2:
                    st.markdown(ai_data.get("root_cause_diagnosis", "No diagnosis available."))

                with subtab3:
                    st.code(ai_data.get("unit_tests", "# No tests generated"), language="python")

    with tab_history:
        st.markdown("#### Test Real Historical Code Updates (from open-source repositories):")
        test_df = load_test_commits()
        if test_df is not None:
            options = [
                f"{'🔴 [Issue Recorded]' if row.label == 1 else '🟢 [Clean Update]'} {str(row.subject)[:60]} ({str(row.date)[:10]})"
                for row in test_df.itertuples()
            ]
            idx = st.selectbox("Select a past update to inspect:", range(min(50, len(options))), format_func=lambda i: options[i])
            row = test_df.iloc[idx]
            
            st.markdown(f"**Description:** `{row['subject']}`")
            st.markdown(f"**Author:** `{row['author']}` &nbsp;|&nbsp; **Date:** `{row['date']}`")
            
            with st.expander("👁️ Show Code Content", expanded=True):
                st.code(row["added_code"] or "(No added lines)", language="python")
            
            if st.button("🔍 Check This Historical Update", type="primary", key="btn_hist_check"):
                dev_values = {f: float(row[f]) for f in DEVELOPER_FEATURES}
                hist_res = predictor.predict(added_code=row["added_code"], removed_code=row["removed_code"], dev_features=dev_values)
                hist_prob = hist_res["hybrid_probability"] * 100
                hist_clean = hist_res["prediction"] == "Clean"
                
                if hist_clean:
                    st.markdown(f"""
                    <div class="status-card-safe">
                        <div class="status-title-safe">🟢 STATUS: SAFE TO SHIP</div>
                        <div class="status-detail-safe"><b>Defect Likelihood:</b> {hist_prob:.0f}% (Low Risk)</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="status-card-risk">
                        <div class="status-title-risk">🔴 STATUS: NEEDS REVIEW</div>
                        <div class="status-detail-risk"><b>Defect Likelihood:</b> {hist_prob:.0f}% (High Risk)</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                reality = "Caused a bug in production" if int(row["label"]) == 1 else "Ran cleanly with no defects"
                st.info(f"**Historical Ground Truth:** In reality, this update *{reality}*.")


if __name__ == "__main__":
    main()
