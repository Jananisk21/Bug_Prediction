"""
app.py
======
Minimal, Intuitive AI Code Safety & Bug Predictor.
- Single-Box Code Checker & File Upload
- Line-by-Line AI Bug Explanation & Fix Suggestions

Run with:  streamlit run app.py
"""

import os
import sys
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.analyzer import analyze_code_lines
from src.features import DEVELOPER_FEATURES
from src.predict import BugPredictor

MODELS_DIR = "models"
TEST_CSV = "data/flask_commits_test.csv"

st.set_page_config(page_title="AI Code Safety Inspector", page_icon="🛡️", layout="centered")


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
    st.markdown("<h1 style='text-align: center;'>🛡️ AI Code Safety Inspector</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: gray; font-size: 1.1rem;'>"
        "Upload or paste any code snippet to detect potential bugs and get line-by-line explanations."
        "</p>",
        unsafe_allow_html=True
    )

    try:
        predictor = load_predictor()
    except Exception as e:
        st.error(f"Could not load AI models: {e}")
        st.stop()

    # Presets & Input Management
    if "input_code" not in st.session_state:
        st.session_state["input_code"] = ""

    tab_input, tab_history = st.tabs(["📝 Check Your Code / File", "📜 Browse Real Historical Samples"])

    with tab_input:
        st.markdown("#### ⚡ 1-Click Quick Examples:")
        c_ex1, c_ex2, c_clear = st.columns([1.2, 1.2, 0.8])
        with c_ex1:
            if st.button("🟢 Clean Code Example", use_container_width=True):
                st.session_state["input_code"] = (
                    "def calculate_discount(price, discount_rate):\n"
                    "    if price is None or price < 0:\n"
                    "        return 0.0\n"
                    "    if discount_rate is None or not (0 <= discount_rate <= 1):\n"
                    "        raise ValueError('Invalid discount rate')\n"
                    "    return round(price * (1.0 - discount_rate), 2)"
                )
        with c_ex2:
            if st.button("🔴 Buggy Code Example", use_container_width=True):
                st.session_state["input_code"] = (
                    "def process_user_order(user_id, order_data):\n"
                    "    try:\n"
                    "        with db.connect() as conn:\n"
                    "            # Dangerous query & unchecked dict access\n"
                    "            conn.execute(f'UPDATE orders SET status = 1 WHERE uid = {user_id}')\n"
                    "            total = order_data['amount'] / order_data['items_count']\n"
                    "            token = auth.get_token(bypass_security=True)\n"
                    "            return total\n"
                    "    except:\n"
                    "        pass\n"
                    "    return None"
                )
        with c_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state["input_code"] = ""

        # File Upload option
        uploaded_file = st.file_uploader("📂 Or drag & drop a code file (.py, .txt):", type=["py", "txt", "js", "java"])
        if uploaded_file is not None:
            try:
                st.session_state["input_code"] = uploaded_file.read().decode("utf-8")
                st.success(f"Loaded `{uploaded_file.name}` successfully!")
            except Exception as e:
                st.error(f"Error reading file: {e}")

        # Single Code Box
        code_text = st.text_area(
            "Paste your code snippet here:",
            value=st.session_state["input_code"],
            height=180,
            placeholder="def my_function():\n    # Paste your code here\n    pass",
            key="main_code_area"
        )

        check_clicked = st.button("🔍 Check Code Safety", type="primary", use_container_width=True)

        if check_clicked:
            if not code_text.strip():
                st.warning("Please enter or upload code first.")
            else:
                # 1. Run Machine Learning Hybrid Prediction
                dev_values = dict(predictor.default_dev_features())
                ml_res = predictor.predict(added_code=code_text, removed_code="", dev_features=dev_values)
                
                # 2. Run Line-by-Line Rule & Pattern Analyzer
                line_issues = analyze_code_lines(code_text)
                
                # If rule-based high severity issues found, elevate probability if needed
                has_high_issues = any(i["severity"] == "High" for i in line_issues)
                risk_prob = ml_res["hybrid_probability"] * 100
                if has_high_issues and risk_prob < 65:
                    risk_prob = 75.0
                    is_clean = False
                else:
                    is_clean = (risk_prob < 50.0) and (len(line_issues) == 0)

                st.markdown("---")
                st.subheader("📋 Safety Assessment Report")

                # Top Status Card
                if is_clean:
                    st.success(
                        f"### 🟢 STATUS: SAFE TO SHIP\n"
                        f"**Defect Likelihood:** **{risk_prob:.0f}%** (Low Risk)\n\n"
                        f"**Action:** ✅ Clean structure. Standard automated testing is sufficient."
                    )
                else:
                    st.error(
                        f"### 🔴 STATUS: NEEDS REVIEW\n"
                        f"**Defect Likelihood:** **{risk_prob:.0f}%** (High Risk)\n\n"
                        f"**Action:** 🚨 Potential bugs or fragile patterns detected. Review recommended before merging."
                    )

                # Line-by-Line AI Explanation
                st.markdown("### 🔍 Line-by-Line AI Findings:")
                if not line_issues:
                    st.info("✅ **No critical line-level code flaws or anti-patterns detected.**")
                else:
                    for issue in line_issues:
                        severity_icon = "🔴" if issue["severity"] == "High" else "🟡"
                        with st.expander(f"{severity_icon} Line {issue['line_no']}: {issue['title']} ({issue['severity']} Risk)", expanded=True):
                            st.code(issue["line_content"], language="python")
                            st.markdown(f"**Why it's risky:** {issue['explanation']}")
                            st.markdown(f"**Suggested fix:** `{issue['recommendation']}`")

                # Code Preview with Line Numbers
                with st.expander("👁️ View Full Code with Line Numbers"):
                    numbered_lines = [f"{i:3d} | {line}" for i, line in enumerate(code_text.splitlines(), start=1)]
                    st.code("\n".join(numbered_lines), language="python")

    with tab_history:
        st.markdown("#### Test Real Historical Code Updates (from a popular open-source project):")
        test_df = load_test_commits()
        if test_df is not None:
            options = [
                f"{'🔴 [Issue Recorded]' if row.label == 1 else '🟢 [Clean Update]'} {str(row.subject)[:60]} ({str(row.date)[:10]})"
                for row in test_df.itertuples()
            ]
            idx = st.selectbox("Select a past update:", range(min(50, len(options))), format_func=lambda i: options[i])
            row = test_df.iloc[idx]
            
            st.markdown(f"**Description:** `{row['subject']}`")
            st.markdown(f"**Author:** `{row['author']}` &nbsp;|&nbsp; **Date:** `{row['date']}`")
            
            with st.expander("Show Code Content", expanded=True):
                st.code(row["added_code"] or "(No added lines)", language="python")
            
            if st.button("🔍 Check This Historical Update", type="primary"):
                dev_values = {f: float(row[f]) for f in DEVELOPER_FEATURES}
                hist_res = predictor.predict(added_code=row["added_code"], removed_code=row["removed_code"], dev_features=dev_values)
                hist_prob = hist_res["hybrid_probability"] * 100
                hist_clean = hist_res["prediction"] == "Clean"
                
                st.markdown("---")
                if hist_clean:
                    st.success(f"### 🟢 STATUS: SAFE TO SHIP ({hist_prob:.0f}% Risk)")
                else:
                    st.error(f"### 🔴 STATUS: NEEDS REVIEW ({hist_prob:.0f}% Risk)")
                
                reality = "Caused a bug in production" if int(row["label"]) == 1 else "Ran cleanly with no defects"
                st.info(f"**Historical Fact:** In reality, this update *{reality}*.")


if __name__ == "__main__":
    main()
