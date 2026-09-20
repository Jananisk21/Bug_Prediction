"""
repo_connector.py
=================
Live repository API connector for fetching commits, PRs, and diffs directly
via GitHub REST API without requiring git CLI commands.
"""

import re
import requests


def parse_repo_url(url: str) -> tuple[str, str]:
    """Extracts owner and repo name from GitHub URL."""
    url = url.strip().rstrip("/")
    match = re.search(r"github\.com[/:]([\w.-]+)/([\w.-]+?)(\.git)?$", url)
    if match:
        return match.group(1), match.group(2)
    # If format is just 'owner/repo'
    parts = url.split("/")
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    raise ValueError(f"Could not parse repository URL: '{url}'. Please use format 'https://github.com/owner/repo'")


def fetch_live_commits(repo_url: str, token: str = None, per_page: int = 20) -> list[dict]:
    """
    Fetches the latest commits from a repository via GitHub REST API.
    Returns list of dicts:
    [{
        'sha': str,
        'short_sha': str,
        'author': str,
        'date': str,
        'message': str,
        'url': str
    }]
    """
    owner, repo = parse_repo_url(repo_url)
    api_url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token and token.strip():
        headers["Authorization"] = f"token {token.strip()}"

    params = {"per_page": per_page}
    resp = requests.get(api_url, headers=headers, params=params, timeout=12)

    if resp.status_code == 404:
        raise ValueError(f"Repository '{owner}/{repo}' not found. Check URL or provide a token for private repos.")
    elif resp.status_code == 403:
        raise ValueError("GitHub API rate limit exceeded. Please provide a Personal Access Token in the token field.")
    elif resp.status_code != 200:
        raise ValueError(f"GitHub API error ({resp.status_code}): {resp.text[:200]}")

    data = resp.json()
    commits = []
    for item in data:
        commit_info = item.get("commit", {})
        author_info = commit_info.get("author", {}) or {}
        sha = item.get("sha", "")
        msg = commit_info.get("message", "").split("\n")[0]
        
        commits.append({
            "sha": sha,
            "short_sha": sha[:8],
            "author": author_info.get("name") or author_info.get("email", "Unknown"),
            "date": author_info.get("date", "")[:10],
            "message": msg,
            "url": item.get("html_url", "")
        })
    return commits


def fetch_commit_diff(repo_url: str, sha: str, token: str = None) -> dict:
    """
    Fetches the detailed code diff and files touched for a specific commit.
    Returns:
    {
        'added_code': str,
        'removed_code': str,
        'files_touched': int,
        'lines_added': int,
        'lines_removed': int,
        'files': list[str]
    }
    """
    owner, repo = parse_repo_url(repo_url)
    api_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
    
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token and token.strip():
        headers["Authorization"] = f"token {token.strip()}"

    resp = requests.get(api_url, headers=headers, timeout=15)
    if resp.status_code != 200:
        raise ValueError(f"Failed to fetch commit {sha[:8]}: ({resp.status_code})")

    data = resp.json()
    files = data.get("files", [])
    
    added_lines = []
    removed_lines = []
    file_names = []
    total_add = 0
    total_del = 0

    for f in files:
        file_names.append(f.get("filename", ""))
        total_add += f.get("additions", 0)
        total_del += f.get("deletions", 0)
        
        patch = f.get("patch", "")
        if patch:
            for line in patch.splitlines():
                if line.startswith("+++") or line.startswith("---"):
                    continue
                if line.startswith("+"):
                    added_lines.append(line[1:])
                elif line.startswith("-"):
                    removed_lines.append(line[1:])

    return {
        "added_code": "\n".join(added_lines),
        "removed_code": "\n".join(removed_lines),
        "files_touched": len(files),
        "lines_added": total_add,
        "lines_removed": total_del,
        "files": file_names
    }
