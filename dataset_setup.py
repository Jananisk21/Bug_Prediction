"""
dataset_setup.py
=================
Builds a real, non-fabricated Just-In-Time (JIT) / change-level bug-prediction
dataset by mining an actual public Git repository.

WHY A MINED DATASET INSTEAD OF A DOWNLOADED ONE
------------------------------------------------
The base review paper (Kamei et al.'s JIT model, DeepJIT, CC2Vec, JITLine)
was trained on the OpenStack/Qt commit datasets originally released by the
DeepJIT/CC2Vec authors. Those releases are distributed as large pickle files
on Google Drive, and the classic six-project Kamei ARFF metrics-only sets
(no diff text) are distributed on Zenodo/Kaggle mirrors. From this
environment, Google Drive, Zenodo, Kaggle and huggingface.co are not
reachable (network policy only allows PyPI/npm-style registries and
github.com/raw.githubusercontent.com). GitHub itself IS reachable, so this
script builds an equivalent dataset directly and transparently: it clones a
real, modest-sized, actively-maintained open-source Python project and mines
it exactly the way the JIT-SDP literature does:

  1. Walk the commit history in chronological order.
  2. Label each commit "bug-inducing" using the classic SZZ algorithm
     (Sliwerski, Zimmermann & Zeller, 2005): find later commits whose message
     indicates a bug fix, then use `git blame` to trace the lines they touch
     back to the commit(s) that last introduced them.
  3. Compute Kamei-style size/diffusion metrics (la, ld, lt, nf, nd, ns,
     entropy, fix) AND developer-activity metrics (EXP, REXP, SEXP, NDEV,
     NUC, AGE, submission rate, contribution interval, recency) using ONLY
     information that existed strictly BEFORE each commit (causal / leakage
     free by construction, not by post-hoc filtering).
  4. Keep the actual added/removed code text of every commit so a code
     embedding can be computed later.

This produces a dataset of the same *shape* and *spirit* as the paper's
target datasets (code diff + developer/process features + bug-inducing
label, time-ordered), built from a real, verifiable, publicly-browsable
commit history -- nothing here is invented or hand-typed.

Default repository: https://github.com/pallets/flask (small, single clean
history, ~5-6k commits -> clones in seconds and mines in a few minutes on a
normal laptop). Any other GitHub repo can be used with --repo-url.

Usage:
    python dataset_setup.py --repo-url https://github.com/pallets/flask.git \
                             --repo-name flask --out-dir data
"""

import argparse
import csv
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

FIX_PATTERN = re.compile(
    r"\b(fix(e[sd])?|bug(s|fix|fixed)?|defect|error|crash(e[sd])?|fault|"
    r"issue|patch|broken|regression|vulnerab\w*|hotfix|incorrect|wrong|"
    r"fail(s|ed|ure)?)\b",
    re.IGNORECASE,
)

# Right-censoring buffer: commits in the final N days of the mined history
# cannot yet be reliably confirmed "clean" (a bug-fix that would prove them
# bug-inducing may simply not have happened yet within the mined window).
# Standard practice in JIT-SDP work is to drop this trailing window rather
# than mislabel it. See README "Leakage & Labeling Caveats".
CENSOR_BUFFER_DAYS = 180

SUBMISSION_WINDOW_DAYS = 90   # trailing window for submission-rate feature
RECENCY_DECAY_DAYS = 30       # decay constant for the recency score

MAX_DIFF_CHARS = 6000         # cap stored diff text (added+removed) per commit


def run(cmd, cwd=None):
    """Run a shell command and return stdout as text (raises on error)."""
    result = subprocess.run(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return result.stdout


def run_ok(cmd, cwd=None):
    """Run a shell command, return (stdout, returncode) without raising."""
    result = subprocess.run(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, errors="replace",
    )
    return result.stdout, result.returncode


def clone_repo(repo_url, repo_dir):
    if os.path.isdir(os.path.join(repo_dir, ".git")):
        print(f"[dataset_setup] Repo already cloned at {repo_dir}, reusing it.")
        return
    print(f"[dataset_setup] Cloning {repo_url} -> {repo_dir} ...")
    subprocess.run(["git", "clone", "--quiet", repo_url, repo_dir], check=True)
    print("[dataset_setup] Clone complete.")


def get_default_branch(repo_dir):
    out = run(["git", "symbolic-ref", "--short", "HEAD"], cwd=repo_dir)
    return out.strip()


def list_commits(repo_dir, branch):
    """All non-merge, single-parent commits reachable from `branch` (i.e.
    every real code-change commit, including those merged in from feature
    branches/PRs -- NOT restricted to --first-parent, which would drop most
    of a GitHub-flow project's actual commits). Returned sorted oldest-first
    by author timestamp, which is what the causal feature pass relies on."""
    fmt = "%H%x1f%P%x1f%ae%x1f%an%x1f%at%x1f%s"
    out = run(
        ["git", "log", "--no-merges", f"--pretty=format:{fmt}", branch],
        cwd=repo_dir,
    )
    commits = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("\x1f")
        if len(parts) != 6:
            continue
        h, parents, ae, an, ts, subj = parts
        parents = parents.strip().split()
        if len(parents) != 1:
            # skip root commit (no parent) - nothing to diff against
            continue
        commits.append(
            {
                "hash": h,
                "parent": parents[0],
                "author_email": ae.strip().lower(),
                "author_name": an.strip(),
                "ts": int(ts),
                "subject": subj,
            }
        )
    commits.sort(key=lambda c: c["ts"])
    return commits


# ----------------------------------------------------------------------------
# Diff parsing: single `git diff -U0 parent commit` gives us everything we
# need -- per-file added/removed line numbers (for SZZ blame targets) and the
# actual added/removed code text (for the embedding).
# ----------------------------------------------------------------------------

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
DIFF_GIT_RE = re.compile(r"^diff --git a/(.*) b/(.*)$")


def get_diff(repo_dir, parent, commit):
    out, rc = run_ok(
        ["git", "diff", "-U0", "--no-color", parent, commit], cwd=repo_dir
    )
    if rc != 0:
        return []
    files = []
    cur = None
    old_path = new_path = None
    is_binary = False
    for line in out.splitlines():
        m = DIFF_GIT_RE.match(line)
        if m:
            if cur is not None and not is_binary:
                files.append(cur)
            old_path, new_path = m.group(1), m.group(2)
            cur = {
                "old_path": old_path,
                "new_path": new_path,
                "added_lines": [],       # list of added code text
                "removed_lines": [],     # list of removed code text
                "del_ranges": [],        # (start,end) 1-based, inclusive, in PARENT file
                "la": 0,
                "ld": 0,
            }
            is_binary = False
            continue
        if line.startswith("Binary files") or line.startswith("GIT binary patch"):
            is_binary = True
            continue
        if cur is None:
            continue
        if line.startswith("--- ") or line.startswith("+++ "):
            continue
        hm = HUNK_RE.match(line)
        if hm:
            old_start, old_len, new_start, new_len = hm.groups()
            old_start = int(old_start)
            old_len = int(old_len) if old_len is not None else 1
            if old_len > 0:
                cur["del_ranges"].append((old_start, old_start + old_len - 1))
            continue
        if line.startswith("-") and not line.startswith("---"):
            cur["removed_lines"].append(line[1:])
            cur["ld"] += 1
        elif line.startswith("+") and not line.startswith("+++"):
            cur["added_lines"].append(line[1:])
            cur["la"] += 1
    if cur is not None and not is_binary:
        files.append(cur)
    return files


def blame_range(repo_dir, revision, path, start, end):
    """Return set of commit hashes that last touched lines [start,end] of
    `path` as of `revision` (i.e. in the parent, before the current commit)."""
    out, rc = run_ok(
        [
            "git", "blame", "--porcelain", "-L", f"{start},{end}",
            revision, "--", path,
        ],
        cwd=repo_dir,
    )
    if rc != 0:
        return set()
    hashes = set()
    for line in out.splitlines():
        # porcelain format: each line-group starts with "<sha> <origline> <finalline> [<numlines>]"
        if re.match(r"^[0-9a-f]{40} ", line):
            hashes.add(line.split()[0])
    return hashes


def subsystem_of(path):
    parts = path.split("/")
    return parts[0] if len(parts) > 1 else "<root>"


def shannon_entropy(counts):
    total = sum(counts)
    if total <= 0 or len(counts) <= 1:
        return 0.0
    ent = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log2(p)
    return ent


def days_between(ts_a, ts_b):
    return (ts_a - ts_b) / 86400.0


# ----------------------------------------------------------------------------
# Main mining pipeline
# ----------------------------------------------------------------------------

def mine(repo_dir, branch, progress_every=500):
    commits = list_commits(repo_dir, branch)
    n = len(commits)
    print(f"[dataset_setup] {n} non-merge mainline commits found.")

    commit_hashes = {c["hash"] for c in commits}

    # ---- Pass 1: parse diffs, run SZZ for fix-commits -----------------
    print("[dataset_setup] Pass 1/2: parsing diffs + SZZ blame for fix commits ...")
    parsed = {}
    bug_inducing = set()
    t0 = time.time()
    n_fix_commits = 0
    for i, c in enumerate(commits):
        files = get_diff(repo_dir, c["parent"], c["hash"])
        parsed[c["hash"]] = files
        is_fix = bool(FIX_PATTERN.search(c["subject"]))
        if is_fix:
            n_fix_commits += 1
            for f in files:
                path = f["old_path"]
                if path == "/dev/null":
                    continue
                for (start, end) in f["del_ranges"]:
                    hashes = blame_range(repo_dir, c["parent"], path, start, end)
                    for h in hashes:
                        if h in commit_hashes and h != c["hash"]:
                            bug_inducing.add(h)
        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(f"    ...{i+1}/{n} commits ({elapsed:.0f}s elapsed)")
    print(
        f"[dataset_setup] Pass 1 done in {time.time()-t0:.0f}s. "
        f"{n_fix_commits} fix-like commits, {len(bug_inducing)} distinct "
        f"commits implicated as bug-inducing by SZZ."
    )

    # ---- Pass 2: causal feature engineering (single chronological pass) ----
    print("[dataset_setup] Pass 2/2: computing causal developer-activity features ...")
    author_ts_history = defaultdict(list)          # author -> [past timestamps]
    author_subsystem_count = defaultdict(int)       # (author,subsystem) -> count
    file_last_ts = {}                                # file -> last change ts
    file_dev_set = defaultdict(set)                  # file -> {authors}
    file_commit_set = defaultdict(set)                # file -> {commit hashes}
    file_line_count = defaultdict(int)                # file -> running LOC estimate

    rows = []
    last_ts = commits[-1]["ts"] if commits else 0

    for c in commits:
        h, ts, author = c["hash"], c["ts"], c["author_email"]
        files = parsed[h]
        touched_new = [f["new_path"] for f in files if f["new_path"] != "/dev/null"]
        touched_old = [f["old_path"] for f in files if f["old_path"] != "/dev/null"]
        touched = list(dict.fromkeys(touched_new + touched_old))  # de-dup, keep order

        # ---- size / diffusion (Kamei-style) ----
        la = sum(f["la"] for f in files)
        ld = sum(f["ld"] for f in files)
        nf = len(touched)
        dirs = {os.path.dirname(p) or "<root>" for p in touched}
        nd = len(dirs)
        subsystems = {subsystem_of(p) for p in touched}
        ns = len(subsystems)
        change_counts = [f["la"] + f["ld"] for f in files]
        entropy = shannon_entropy(change_counts)
        lt_before = sum(file_line_count.get(p, 0) for p in touched)
        fix = int(bool(FIX_PATTERN.search(c["subject"])))

        # ---- developer / process features (CAUSAL: uses only prior state) ----
        exp = len(author_ts_history[author])
        if exp > 0:
            rexp = sum(
                1.0 / (max(days_between(ts, t_prev), 0) / 365.25 + 1.0)
                for t_prev in author_ts_history[author]
            )
        else:
            rexp = 0.0
        sexp = sum(author_subsystem_count[(author, s)] for s in subsystems)

        ndev_set = set()
        nuc_set = set()
        ages = []
        for p in touched:
            ndev_set |= file_dev_set.get(p, set())
            nuc_set |= file_commit_set.get(p, set())
            if p in file_last_ts:
                ages.append(days_between(ts, file_last_ts[p]))
        ndev = len(ndev_set)
        nuc = len(nuc_set)
        age = (sum(ages) / len(ages)) if ages else 0.0

        window_start = ts - SUBMISSION_WINDOW_DAYS * 86400
        n_recent = sum(1 for t_prev in author_ts_history[author] if t_prev >= window_start)
        submission_rate = n_recent / SUBMISSION_WINDOW_DAYS

        if author_ts_history[author]:
            last_author_ts = author_ts_history[author][-1]
            contribution_interval = days_between(ts, last_author_ts)
            recency = math.exp(-max(contribution_interval, 0) / RECENCY_DECAY_DAYS)
        else:
            contribution_interval = -1.0   # sentinel: first-ever commit by this author
            recency = 0.0

        # ---- label ----
        days_from_end = days_between(last_ts, ts)
        censored = days_from_end < CENSOR_BUFFER_DAYS
        label = 1 if h in bug_inducing else 0

        # ---- diff text for the code branch ----
        added_txt = "\n".join(l for f in files for l in f["added_lines"])
        removed_txt = "\n".join(l for f in files for l in f["removed_lines"])
        added_txt = added_txt[:MAX_DIFF_CHARS]
        removed_txt = removed_txt[:MAX_DIFF_CHARS]

        rows.append(
            {
                "commit_hash": h,
                "timestamp": ts,
                "date": datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "author": author,
                "subject": c["subject"][:200],
                "la": la, "ld": ld, "lt": lt_before, "nf": nf, "nd": nd, "ns": ns,
                "entropy": round(entropy, 4), "fix": fix,
                "exp": exp, "rexp": round(rexp, 4), "sexp": sexp, "ndev": ndev,
                "nuc": nuc, "age": round(age, 4),
                "submission_rate": round(submission_rate, 5),
                "contribution_interval": round(contribution_interval, 4),
                "recency": round(recency, 4),
                "censored": int(censored),
                "label": label,
                "added_code": added_txt,
                "removed_code": removed_txt,
                "files_touched": ";".join(touched)[:500],
            }
        )

        # ---- update causal state AFTER computing this row's features ----
        author_ts_history[author].append(ts)
        for s in subsystems:
            author_subsystem_count[(author, s)] += 1
        for f in files:
            p_new = f["new_path"]
            p_old = f["old_path"]
            if p_new != "/dev/null":
                file_last_ts[p_new] = ts
                file_dev_set[p_new].add(author)
                file_commit_set[p_new].add(h)
                file_line_count[p_new] = max(0, file_line_count.get(p_old, 0) + f["la"] - f["ld"])
            if p_old != "/dev/null" and p_old != p_new:
                file_line_count[p_old] = 0  # renamed/removed, old path no longer exists

    print(f"[dataset_setup] Pass 2 done. {len(rows)} rows total.")
    return rows


def write_csv(rows, path):
    if not rows:
        print("[dataset_setup] No rows to write!")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[dataset_setup] Wrote {len(rows)} rows -> {path}")


def time_based_split(rows, test_frac=0.2):
    """Drop right-censored tail, then split the remainder chronologically:
    the earliest (1-test_frac) commits -> train, the most recent -> test.
    No shuffling anywhere -- this is a strict temporal holdout."""
    usable = [r for r in rows if r["censored"] == 0]
    usable.sort(key=lambda r: r["timestamp"])
    n = len(usable)
    split_idx = int(n * (1 - test_frac))
    train = usable[:split_idx]
    test = usable[split_idx:]
    return train, test


def summarize(rows, name):
    n = len(rows)
    pos = sum(r["label"] for r in rows)
    print(f"    {name}: n={n}, bug-inducing={pos} ({100*pos/max(n,1):.1f}%)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-url", default="https://github.com/pallets/flask.git")
    ap.add_argument("--repo-name", default="flask")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    repo_dir = os.path.join(args.out_dir, "_repo_" + args.repo_name)

    clone_repo(args.repo_url, repo_dir)
    branch = get_default_branch(repo_dir)
    print(f"[dataset_setup] Default branch: {branch}")

    rows = mine(repo_dir, branch)

    all_path = os.path.join(args.out_dir, f"{args.repo_name}_commits_all.csv")
    write_csv(rows, all_path)

    train, test = time_based_split(rows, test_frac=args.test_frac)
    write_csv(train, os.path.join(args.out_dir, f"{args.repo_name}_commits_train.csv"))
    write_csv(test, os.path.join(args.out_dir, f"{args.repo_name}_commits_test.csv"))

    print("\n[dataset_setup] ===== Summary =====")
    print(f"    Repository: {args.repo_url}")
    n_censored = sum(r["censored"] for r in rows)
    print(f"    Total mined commits: {len(rows)} (of which {n_censored} excluded as right-censored, "
          f"i.e. within the last {CENSOR_BUFFER_DAYS} days of mined history)")
    summarize(train, "Train (older commits)")
    summarize(test, "Test  (newer commits)")
    print("[dataset_setup] Done.")


if __name__ == "__main__":
    main()
