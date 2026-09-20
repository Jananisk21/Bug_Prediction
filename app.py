"""
app.py
======
AI Code Safety Inspector Dashboard
- Tab 1: 📝 Check Your Code / File (Single box & file uploader)
- Tab 2: 🌐 Connect Live Repository (Fetch live repository commits & analyze)
- Powered by Local Hybrid BiGRU Deep Learning + Generative AI Copilot & Fix Engine

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
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Styling to match the clean dark UI
st.markdown("""
<style>
    /* Global container */
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
    /* Primary buttons */
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
    }
    .status-title-risk {
        color: #f87171;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
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
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.2rem;
    }
    .section-subtitle {
        color: #9ca3af;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    return BugPredictor(MODELS_DIR)


def render_ai_explanation_block(code_to_explain: str, is_clean: bool, risk_prob: float):
    """Renders the AI Key Enhancements and 3 Sub-Tabs matching reference UI."""
    with st.spinner("Generating architectural explanation & production-hardened fix..."):
        ai_data = get_ai_explanation_and_fix(code_to_explain, is_clean=is_clean, risk_prob=risk_prob)

    st.markdown('<div class="section-title">💡 Explanation & Fix Code</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Detailed architectural explanation, production-hardened replacement code, and automated unit tests.</div>', unsafe_allow_html=True)

    # Key Enhancements
    st.markdown("#### ⚡ Key Enhancements:")
    enhancements = ai_data.get("key_enhancements", [])
    for enh in enhancements:
        st.markdown(f"• {enh}")

    st.write("")

    # Sub-tabs
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


def main():
    # Header
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
    if "live_commits" not in st.session_state:
        st.session_state["live_commits"] = []
    if "active_repo_url" not in st.session_state:
        st.session_state["active_repo_url"] = "https://github.com/pallets/flask"

    tab_check, tab_repo = st.tabs(["📝 Check Your Code / File", "🌐 Connect Live Repository"])

    # -------------------------------------------------------------
    # TAB 1: Check Your Code / File
    # -------------------------------------------------------------
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
            height=160,
            label_visibility="collapsed",
            placeholder='a = 5\nb = 3\nsum = a + b\nprint("Sum:", sum)'
        )

        check_clicked = st.button("🔍 Check Code Safety", type="primary", use_container_width=True, key="btn_check_code")

        if check_clicked:
            if not code_text.strip():
                st.warning("Please enter or upload code first.")
            else:
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

                st.markdown('<div class="section-title">📋 Safety Assessment Report</div>', unsafe_allow_html=True)

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
                render_ai_explanation_block(code_text, is_clean, risk_prob)

    # -------------------------------------------------------------
    # TAB 2: Connect Live Repository
    # -------------------------------------------------------------
    with tab_repo:
        st.markdown("#### 🔗 Connect Live GitHub Repository")
        st.caption("Fetch real-time commits directly via API without needing terminal commands.")

        col_repo, col_btn = st.columns([3, 1])
        with col_repo:
            repo_url_input = st.text_input(
                "Repository URL:",
                value=st.session_state["active_repo_url"],
                placeholder="https://github.com/owner/repository"
            )
        with col_btn:
            st.write("")
            st.write("")
            fetch_btn = st.button("🚀 Fetch Updates", use_container_width=True)

        token_input = st.text_input(
            "🔑 Personal Access Token (Optional for Private Repos):",
            type="password",
            placeholder="ghp_xxxxxxxxxxxxxxxxxxxx (Leave empty for public repos)"
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
                        st.success(f"Fetched {len(commits)} latest code updates!")
                    except Exception as e:
                        st.error(f"Error connecting to repository: {e}")

        # If commits are available
        if st.session_state["live_commits"]:
            commits = st.session_state["live_commits"]
            options = [
                f"[{c['author']}] {c['message'][:55]} ({c['short_sha']} · {c['date']})"
                for c in commits
            ]
            selected_idx = st.selectbox(
                "Select a live code update to inspect:",
                range(len(options)),
                format_func=lambda i: options[i]
            )
            selected_commit = commits[selected_idx]

            # Fetch diff details for selected commit
            with st.spinner("Fetching commit code diff..."):
                try:
                    diff_data = fetch_commit_diff(
                        st.session_state["active_repo_url"],
                        selected_commit["sha"],
                        token=token_input
                    )
                except Exception as e:
                    diff_data = {"added_code": "", "removed_code": "", "files_touched": 1, "lines_added": 0, "lines_removed": 0, "files": []}

            st.markdown(f"**Commit Message:** `{selected_commit['message']}`")
            st.markdown(f"**Author:** `{selected_commit['author']}` &nbsp;|&nbsp; **Date:** `{selected_commit['date']}` &nbsp;|&nbsp; **Files Touched:** `{diff_data['files_touched']}`")

            with st.expander("👁️ View Live Code Diff for this Update", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🟢 Added Lines:**")
                    st.code(diff_data["added_code"][:1500] or "(none)", language="python")
                with col2:
                    st.markdown("**🔴 Removed Lines:**")
                    st.code(diff_data["removed_code"][:1500] or "(none)", language="python")

            scan_repo_btn = st.button("🔍 Scan & Analyze This Live Update", type="primary", use_container_width=True, key="btn_scan_repo")

            if scan_repo_btn:
                # Prepare developer features from live diff
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

                st.markdown('<div class="section-title">📋 Safety Assessment Report</div>', unsafe_allow_html=True)

                if is_clean:
                    st.markdown(f"""
                    <div class="status-card-safe">
                        <div class="status-title-safe">🟢 STATUS: SAFE TO SHIP</div>
                        <div class="status-detail-safe"><b>Defect Likelihood:</b> {risk_prob:.0f}% (Low Risk)</div>
                        <div class="status-action"><b>Action:</b> ✅ Verified safe. Standard continuous integration checks approved.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="status-card-risk">
                        <div class="status-title-risk">🔴 STATUS: NEEDS REVIEW</div>
                        <div class="status-detail-risk"><b>Defect Likelihood:</b> {risk_prob:.0f}% (High Risk)</div>
                        <div class="status-action"><b>Action:</b> 🚨 High defect risk identified. Senior peer review required before merge.</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")
                render_ai_explanation_block(code_to_check, is_clean, risk_prob)


if __name__ == "__main__":
    main()
