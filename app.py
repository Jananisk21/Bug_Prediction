"""
app.py
======
AI Code Safety Inspector Dashboard - Side-by-Side (Split-Screen) Layout
- Left Column: Code Inputs, File Uploader, and Live Repository Connector
- Right Column: Real-time Safety Assessment, AI Key Enhancements, Refactored Code, Root-Cause Diagnosis, and Pytest Test Suites

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
from src.repo_connector import fetch_live_commits, fetch_commit_diff

MODELS_DIR = "models"

st.set_page_config(
    page_title="AI Code Safety Inspector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark modern split-view styling
st.markdown("""
<style>
    /* Full width styling with comfortable padding */
    .block-container {
        max-width: 96%;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .header-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .header-subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    /* Buttons */
    div.stButton > button:first-child {
        background-color: #ff4b4b;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.55rem 1.2rem;
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
        padding: 1.1rem 1.3rem;
        margin-top: 0.5rem;
        margin-bottom: 1.2rem;
    }
    .status-card-risk {
        background-color: #381212;
        border: 1px solid #ef4444;
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        margin-top: 0.5rem;
        margin-bottom: 1.2rem;
    }
    .status-title-safe {
        color: #34d399;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .status-title-risk {
        color: #f87171;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .status-detail-safe {
        color: #6ee7b7;
        font-size: 0.95rem;
        margin-bottom: 0.3rem;
    }
    .status-detail-risk {
        color: #fca5a5;
        font-size: 0.95rem;
        margin-bottom: 0.3rem;
    }
    .status-action {
        color: #d1d5db;
        font-size: 0.92rem;
    }
    .placeholder-card {
        background-color: #161b26;
        border: 1px dashed #374151;
        border-radius: 10px;
        padding: 3.5rem 2rem;
        text-align: center;
        color: #9ca3af;
        margin-top: 1rem;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.2rem;
    }
    .section-subtitle {
        color: #9ca3af;
        font-size: 0.9rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    return BugPredictor(MODELS_DIR)


def main():
    # Header
    st.markdown('<div class="header-title">🛡️ AI Code Safety Inspector</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Detect potential bugs, analyze risk likelihood, and generate automated explanations with fixed code.</div>', unsafe_allow_html=True)

    try:
        predictor = load_predictor()
    except Exception as e:
        st.error(f"Could not load AI models: {e}")
        st.stop()

    # Session State Initialization
    if "code_input" not in st.session_state:
        st.session_state["code_input"] = 'a = 5\nb = 3\nsum = a + b\nprint("Sum:", sum)'
    if "live_commits" not in st.session_state:
        st.session_state["live_commits"] = []
    if "active_repo_url" not in st.session_state:
        st.session_state["active_repo_url"] = "https://github.com/pallets/flask"
    if "analysis_result" not in st.session_state:
        st.session_state["analysis_result"] = None

    # Side-by-Side Split Columns
    col_left, col_right = st.columns([1, 1.15], gap="large")

    # =========================================================================
    # 👈 LEFT COLUMN: Inputs & Controls
    # =========================================================================
    with col_left:
        tab_check, tab_repo = st.tabs(["📝 Check Your Code / File", "🌐 Connect Live Repository"])

        # Tab 1: Check Code / File
        with tab_check:
            st.markdown("📂 **Upload or drag & drop a code file (.py, .txt):**")
            uploaded_file = st.file_uploader(
                "Upload",
                type=["py", "txt", "js", "java"],
                label_visibility="collapsed"
            )
            if uploaded_file is not None:
                try:
                    st.session_state["code_input"] = uploaded_file.read().decode("utf-8")
                    st.success(f"Loaded `{uploaded_file.name}` successfully!")
                except Exception as e:
                    st.error(f"Error reading file: {e}")

            st.markdown("Paste your code snippet here:")
            code_text = st.text_area(
                "Paste your code snippet here:",
                value=st.session_state["code_input"],
                height=180,
                label_visibility="collapsed",
                placeholder='a = 5\nb = 3\nsum = a + b\nprint("Sum:", sum)'
            )

            if st.button("🔍 Check Code Safety", type="primary", use_container_width=True, key="btn_check_code"):
                if not code_text.strip():
                    st.warning("Please enter or upload code first.")
                else:
                    with st.spinner("Analyzing code safety..."):
                        dev_values = dict(predictor.default_dev_features())
                        ml_res = predictor.predict(added_code=code_text, removed_code="", dev_features=dev_values)
                        
                        line_issues = analyze_code_lines(code_text)
                        has_high_issues = any(i["severity"] == "High" for i in line_issues)
                        
                        risk_prob = ml_res["hybrid_probability"] * 100
                        if not has_high_issues and risk_prob < 15.0:
                            risk_prob = 5.0
                            is_clean = True
                        elif has_high_issues:
                            risk_prob = max(risk_prob, 76.0)
                            is_clean = False
                        else:
                            is_clean = (risk_prob < 50.0)

                        # Generate AI Fix Suite
                        ai_data = get_ai_explanation_and_fix(code_text, is_clean=is_clean, risk_prob=risk_prob)

                        st.session_state["analysis_result"] = {
                            "is_clean": is_clean,
                            "risk_prob": risk_prob,
                            "code_source": "custom",
                            "ai_data": ai_data
                        }

        # Tab 2: Connect Live Repository
        with tab_repo:
            st.markdown("#### 🔗 Live Repository Connector")
            st.caption("Fetch real-time commits directly via GitHub REST API.")

            c_url, c_btn = st.columns([2.5, 1])
            with c_url:
                repo_url_input = st.text_input(
                    "Repository URL:",
                    value=st.session_state["active_repo_url"],
                    placeholder="https://github.com/owner/repository"
                )
            with c_btn:
                st.write("")
                st.write("")
                fetch_btn = st.button("🚀 Fetch", use_container_width=True)

            token_input = st.text_input(
                "🔑 Token (Optional for Private Repos):",
                type="password",
                placeholder="ghp_xxxxxxxxxxxx (Leave empty for public)"
            )

            if fetch_btn:
                if not repo_url_input.strip():
                    st.warning("Please enter a valid repository URL.")
                else:
                    with st.spinner("Connecting to repository API..."):
                        try:
                            commits = fetch_live_commits(repo_url_input, token=token_input, per_page=15)
                            st.session_state["live_commits"] = commits
                            st.session_state["active_repo_url"] = repo_url_input
                            st.success(f"Fetched {len(commits)} latest updates!")
                        except Exception as e:
                            st.error(f"Error connecting: {e}")

            if st.session_state["live_commits"]:
                commits = st.session_state["live_commits"]
                options = [
                    f"[{c['author']}] {c['message'][:45]} ({c['short_sha']} · {c['date']})"
                    for c in commits
                ]
                selected_idx = st.selectbox(
                    "Select a commit update to inspect:",
                    range(len(options)),
                    format_func=lambda i: options[i]
                )
                selected_commit = commits[selected_idx]

                # Fetch diff
                try:
                    diff_data = fetch_commit_diff(
                        st.session_state["active_repo_url"],
                        selected_commit["sha"],
                        token=token_input
                    )
                except Exception:
                    diff_data = {"added_code": "", "removed_code": "", "files_touched": 1, "lines_added": 0, "lines_removed": 0}

                st.markdown(f"**Author:** `{selected_commit['author']}` &nbsp;|&nbsp; **Date:** `{selected_commit['date']}` &nbsp;|&nbsp; **Files:** `{diff_data['files_touched']}`")

                with st.expander("👁️ View Live Code Diff", expanded=False):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**🟢 Added:**")
                        st.code(diff_data["added_code"][:1000] or "(none)", language="python")
                    with c2:
                        st.markdown("**🔴 Removed:**")
                        st.code(diff_data["removed_code"][:1000] or "(none)", language="python")

                if st.button("🔍 Scan & Analyze This Live Update", type="primary", use_container_width=True, key="btn_scan_repo"):
                    with st.spinner("Analyzing live commit with AI..."):
                        dev_values = dict(predictor.default_dev_features())
                        dev_values["nf"] = float(diff_data["files_touched"])
                        dev_values["la"] = float(diff_data["lines_added"])
                        dev_values["ld"] = float(diff_data["lines_removed"])

                        code_to_check = diff_data["added_code"] if diff_data["added_code"].strip() else selected_commit["message"]

                        ml_res = predictor.predict(
                            added_code=diff_data["added_code"],
                            removed_code=diff_data["removed_code"],
                            dev_features=dev_values
                        )
                        risk_prob = ml_res["hybrid_probability"] * 100
                        is_clean = (risk_prob < 50.0)

                        ai_data = get_ai_explanation_and_fix(code_to_check, is_clean=is_clean, risk_prob=risk_prob)

                        st.session_state["analysis_result"] = {
                            "is_clean": is_clean,
                            "risk_prob": risk_prob,
                            "code_source": "repo",
                            "ai_data": ai_data
                        }

    # =========================================================================
    # 👉 RIGHT COLUMN: Live Results & AI Inspector
    # =========================================================================
    with col_right:
        if st.session_state["analysis_result"] is None:
            st.markdown("""
            <div class="placeholder-card">
                <h3>👈 Ready for Inspection</h3>
                <p>Paste code, upload a file, or connect a live repository on the left, then click <b>'Check Code Safety'</b> to see the real-time AI Safety Report here.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            res = st.session_state["analysis_result"]
            is_clean = res["is_clean"]
            risk_prob = res["risk_prob"]
            ai_data = res["ai_data"]

            st.markdown('<div class="section-title">📋 Safety Assessment Report</div>', unsafe_allow_html=True)

            # Status Banner
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
                    <div class="status-action"><b>Action:</b> 🚨 Potential bugs or fragile patterns detected. Review required before merging.</div>
                </div>
                """, unsafe_allow_html=True)

            # AI Explanation & Fix Code Section
            st.markdown('<div class="section-title">💡 Explanation & Fix Code</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-subtitle">Detailed architectural explanation, production-hardened replacement code, and automated unit tests.</div>', unsafe_allow_html=True)

            # Key Enhancements
            st.markdown("#### ⚡ Key Enhancements:")
            enhancements = ai_data.get("key_enhancements", [])
            for enh in enhancements:
                st.markdown(f"• {enh}")

            st.write("")

            # 3 Sub-tabs
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


if __name__ == "__main__":
    main()
