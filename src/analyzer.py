"""
analyzer.py
===========
Line-by-line code safety & risk analyzer.
Scans Python code for bug-prone patterns, anti-patterns, and vulnerability indicators,
providing plain-English explanations and actionable fix suggestions for non-technical users.
"""

import ast
import re


def analyze_code_lines(code: str) -> list[dict]:
    """
    Analyzes code line-by-line for common bug-inducing patterns.
    Returns a list of issue dictionaries:
    [{
        'line_no': int,
        'line_content': str,
        'severity': 'High' | 'Medium' | 'Low',
        'title': str,
        'explanation': str,
        'recommendation': str
    }]
    """
    if not code or not code.strip():
        return []

    lines = code.splitlines()
    issues = []

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()

        # 1. Bare except or silent exception suppression
        if re.search(r"except\s*:\s*$", stripped) or re.search(r"except\s+\w+\s*:\s*(pass|return None)\s*$", stripped):
            issues.append({
                "line_no": idx,
                "line_content": stripped,
                "severity": "High",
                "title": "Silent Error Catching",
                "explanation": "This code catches all errors and silently ignores them (using 'pass' or bare 'except:'). This hides real bugs and makes troubleshooting impossible.",
                "recommendation": "Catch specific exceptions (e.g. 'except ValueError:') and log the error message."
            })

        # 2. Raw SQL string formatting / SQL Injection risk
        elif re.search(r"\bexecute\s*\(\s*f[\"'].*[\"']", stripped) or re.search(r"\bexecute\s*\(\s*[\"'].*%s.*[\"']\s*%", stripped):
            issues.append({
                "line_no": idx,
                "line_content": stripped,
                "severity": "High",
                "title": "Unsafe Database Query (SQL Injection)",
                "explanation": "Variables are directly concatenated or formatted into a database query string instead of using parameterized queries.",
                "recommendation": "Use parameterized queries, e.g., conn.execute('SELECT * FROM t WHERE id = ?', (id,))."
            })

        # 3. Security bypass or hardcoded sensitive keywords
        elif re.search(r"\b(bypass_security|verify\s*=\s*False|allow_insecure\s*=\s*True|ignore_auth\s*=\s*True)\b", stripped):
            issues.append({
                "line_no": idx,
                "line_content": stripped,
                "severity": "High",
                "title": "Security Bypass Detected",
                "explanation": "A flag explicitly disabling authentication, SSL verification, or security checks was found.",
                "recommendation": "Ensure security checks remain active in production environments."
            })

        # 4. Division without zero-check
        elif re.search(r"\w+\s*/\s*[a-zA-Z_]\w*", stripped) and not re.search(r"if\s+.*!=\s*0", stripped):
            if not stripped.startswith("#") and not stripped.startswith("import") and not stripped.startswith("from"):
                issues.append({
                    "line_no": idx,
                    "line_content": stripped,
                    "severity": "Medium",
                    "title": "Potential Division by Zero",
                    "explanation": "A division operation is performed with a dynamic variable divisor without verifying that it is non-zero.",
                    "recommendation": "Add a check before dividing, e.g.: 'if divisor != 0: ...'"
                })

        # 5. File / connection opened without 'with' statement
        elif re.search(r"\bopen\s*\(", stripped) and not stripped.startswith("with ") and not stripped.startswith("#"):
            issues.append({
                "line_no": idx,
                "line_content": stripped,
                "severity": "Medium",
                "title": "Resource Leak Risk",
                "explanation": "A file or resource is opened directly without a 'with' context manager. If an error occurs, the file may remain open in memory.",
                "recommendation": "Use context managers: 'with open(...) as f:' to ensure automatic closure."
            })

        # 6. Mutable default arguments in function definitions
        elif re.search(r"def\s+\w+\s*\(.*=\s*(\[\]|\{\})\s*\):", stripped):
            issues.append({
                "line_no": idx,
                "line_content": stripped,
                "severity": "Medium",
                "title": "Mutable Default Argument",
                "explanation": "Using a mutable list '[]' or dict '{}' as a default parameter causes that list to be shared across all function calls.",
                "recommendation": "Use 'None' as the default argument and initialize the list inside the function."
            })

        # 7. Unchecked dictionary or object access
        elif re.search(r"\b(data|response|payload|item|config)\[[\"']\w+[\"']\]", stripped) and "if " not in stripped and "try" not in stripped:
            if not stripped.startswith("#"):
                issues.append({
                    "line_no": idx,
                    "line_content": stripped,
                    "severity": "Low",
                    "title": "Unchecked Key Access",
                    "explanation": "Accessing dictionary keys directly (e.g. data['key']) will crash with a KeyError if the key is missing.",
                    "recommendation": "Use '.get()' method, e.g. data.get('key', default_value)."
                })

    return issues
