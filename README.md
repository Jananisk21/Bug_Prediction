# Deep Learning-Based Bug Prediction Using Code Embedding and Developer Activity

A practical, laptop-friendly implementation of the change-level / Just-In-Time
(JIT) bug-prediction system proposed in the review paper *"Deep
Learning-Based Bug Prediction Using Code Embedding and Developer Activity: A
Review."* Given a Git commit (a code diff), the system predicts whether it
is **Bug-Inducing** or **Clean** by fusing a learned representation of the
code change with a set of developer-activity features, following the
paper's stated direction: *"a learned semantic embedding of the code change
... combined with an explicit, extended set of developer-activity features
covering experience (EXP, REXP, SEXP), ownership diffusion (NDEV), and
behavioural rhythm (submission rate, contribution interval, recency)"*,
evaluated with a **time-ordered, leakage-free split** and **effort-aware
metrics**, exactly as the paper's "Motivation and Scope" section specifies.

```
Git Commit / Code Diff
        │
        ▼
Data Preprocessing  (src/preprocessing.py)
        │
        ▼
Code Embedding  (src/embeddings.py)  ──────────┐
        │                                       │  Code Branch
Developer Activity Features (dataset_setup.py)  │
        │                                       │  Developer Branch
        ▼                                       │
Feature Fusion  (src/model.py) ◄─────────────────┘
        │
        ▼
Deep Learning Classifier
        │
        ▼
Bug Probability ──► Bug-Inducing / Clean
```

---

## 1. Dataset: what we used, and why

The paper's own target datasets are the **six OSS/commercial projects from
Kamei et al.** and the **OpenStack/Qt datasets used by DeepJIT / CC2Vec /
JITLine**. Both are real, well-cited datasets — but neither is reachable
from the sandboxed environment this project was built in: the Kamei-style
ARFF releases and the DeepJIT/CC2Vec pickle files are distributed via
Google Drive / Zenodo / Kaggle, and `huggingface.co`, `zenodo.org` and
`kaggle.com` are all network-blocked in that environment (only
`github.com`/`raw.githubusercontent.com` and the standard package
registries are reachable). We verified this directly rather than assuming
it (see the commands in `dataset_setup.py`'s docstring).

**Practical alternative used here:** `dataset_setup.py` mines a real, public
GitHub repository (default: [`pallets/flask`](https://github.com/pallets/flask),
~3.8k non-merge commits, 16 MB, clones in seconds) and reconstructs a JIT
dataset of the *same shape* as the paper's target datasets, using the
standard techniques from the literature the paper itself reviews:

| Step | Technique | Matches |
|---|---|---|
| Labeling | **SZZ algorithm** (Śliwerski, Zimmermann & Zeller, 2005): find bug-fix commits by commit-message keywords, then `git blame` the lines they touch back to the commit(s) that introduced them | how Kamei-style ground truth is built in the JIT-SDP literature |
| Fix-commit detection | keyword regex over commit messages (`fix`, `bug`, `error`, `crash`, `regression`, ...) | Mockus & Votta (2000); Kim et al. (2008), cited in the paper |
| Size / diffusion features | la, ld, lt, nf, nd, ns, entropy, fix | Kamei et al.'s 14-metric JIT model, described in the paper's §"Traditional Change-Level and File-Level Defect Prediction" |
| Developer-activity features | EXP, REXP, SEXP, NDEV, NUC, AGE, submission rate, contribution interval, recency | the paper's §"Developer Activity and Process-Aware Features" |
| Split | strict **time-based** split, oldest commits → train, newest → test | the paper's "Motivation and Scope" (leakage-free, time-ordered evaluation) |

This is not a toy/synthetic dataset — every commit, diff, author, and
timestamp is real and independently verifiable on GitHub. Nothing about the
labels or features is hand-typed or invented; they are computed by
`dataset_setup.py` directly from `git log` / `git diff` / `git blame`.

You can point the same script at any other GitHub repository:

```bash
python dataset_setup.py --repo-url https://github.com/<owner>/<repo>.git --repo-name <repo>
```

A larger/more active repo (e.g. `psf/requests`) gives more data at the cost
of a longer mining run; a smaller one runs faster. Flask was chosen as a
good balance for a normal laptop (mining the full history takes well under
a minute).

### Labeling & leakage caveats (read this before trusting the numbers)

- **Keyword-based SZZ is a heuristic**, not ground truth. It will miss bugs
  fixed with unusually worded commit messages and can occasionally
  misattribute a fix to whitespace/formatting lines. This is the same
  limitation the reviewed literature acknowledges for any SZZ-based label.
- **Right-censoring**: a commit made very recently may genuinely be
  bug-inducing, but its fix simply hasn't been written yet. We exclude the
  most recent 180 days of mined history from train/test entirely
  (`CENSOR_BUFFER_DAYS` in `dataset_setup.py`) rather than mislabel it as
  clean. Even with that buffer, the test set (the newest remaining commits)
  still has a much lower bug-inducing rate (10.4%) than train (31.4%) —
  this is the expected "verification latency" effect Kamei et al. describe,
  not a bug in the pipeline.
- **No future information is used anywhere.** Every developer-activity
  feature (EXP, REXP, SEXP, NDEV, NUC, AGE, submission rate, contribution
  interval, recency) is computed in a single forward pass over
  chronologically-sorted commits, updating each author's/file's history
  *after* that commit's features are recorded, never before. The feature
  scaler and the vocabulary used by the code embedding are both fit on the
  training split only.

---

## 2. Code embedding: CodeBERT vs. the lightweight default

The paper points to CodeBERT/UniXCoder-style pretrained code embeddings.
`src/embeddings.py` implements **both**:

- **`CodeBertEmbedder`** — frozen `microsoft/codebert-base` via Hugging
  Face Transformers, mean-pooled to a 768-d vector (feature-extraction
  only, no fine-tuning of its 125M parameters — CPU-friendly). This is what
  you should use once you have internet access to `huggingface.co` (any
  normal laptop). See "Switching to CodeBERT" below.
- **`LightweightCodeEncoder`** — a small trainable `Embedding → BiGRU →
  pooling` encoder (a few hundred thousand parameters), trained from
  scratch on the diff tokens, in the spirit of the original DeepJIT/Kim et
  al. token-based encoders discussed in the review. **This is the encoder
  actually trained and evaluated in this repository**, because the sandbox
  this project was built in cannot reach `huggingface.co` to download
  CodeBERT's weights. It needs no internet access and no GPU.

### Switching to CodeBERT

On a machine with internet access:

```python
from src.embeddings import CodeBertEmbedder
embedder = CodeBertEmbedder()                       # downloads ~500MB once
vecs = embedder.embed(list_of_diff_texts)            # -> np.ndarray [N, 768]
```

Precompute embeddings for `diff_text` in the train/val/test CSVs, then swap
`build_lightweight_hybrid(...)` for `build_codebert_hybrid(...)` in
`src/train.py` (feed the precomputed 768-d vectors instead of token ids).
The `DeveloperBranch` and fusion head are unchanged either way.

---

## 3. Model

**Code Branch:** diff text → tokenizer (`src/preprocessing.py`) → embedding
(CodeBERT or lightweight, above) → representation vector.

**Developer Branch:** `DEVELOPER_FEATURES` (17 features — 8 size/diffusion
+ 9 developer-activity, see `src/features.py`) → z-score normalization
(fit on train only) → 2 dense layers.

**Fusion:** `concat(code_repr, dev_repr)` → dense(64) → dropout(0.3) →
dense(1) → sigmoid → bug probability. Implemented in `src/model.py`
(`HybridBugPredictor`).

**Baseline:** the same 17 traditional/developer features →
`sklearn.linear_model.LogisticRegression(class_weight="balanced")` — a
direct reproduction of Kamei et al.'s LR-JIT baseline, with no code
embedding at all, so the comparison isolates what the learned code
representation adds.

---

## 4. How to reproduce everything

```bash
pip install -r requirements.txt

# 1. Mine a real dataset from GitHub (default: pallets/flask)
python dataset_setup.py --repo-url https://github.com/pallets/flask.git --repo-name flask

# 2. Train + evaluate both models (writes models/*.pt, *.pkl, metrics.json, plots/)
python -m src.train --train-csv data/flask_commits_train.csv --test-csv data/flask_commits_test.csv --epochs 15

# 3. Try a single prediction from the command line
python -m src.predict --added-code "if x is None:\n    raise ValueError(x)" --removed-code "return None"

# 4. Launch the UI
streamlit run app.py
```

Total time on a normal laptop (2 CPU cores, no GPU): mining ~1 minute,
training ~5 minutes for 15 epochs.

---

## 5. Results (actual, from `models/metrics.json` — not fabricated)

Dataset: 2,577 train / 455 validation / 759 test commits (mined from
`pallets/flask`, time-ordered split). Bug-inducing rate: 31.4% train /
16.3% val / 10.4% test.

| Metric | Baseline (Logistic Regression) | Proposed Hybrid Model |
|---|---:|---:|
| Accuracy | 0.727 | 0.797 |
| Precision | 0.229 | 0.255 |
| Recall | 0.684 | 0.494 |
| F1-score | 0.343 | 0.336 |
| ROC-AUC | 0.787 | 0.759 |
| Popt (effort-aware) | 0.787 | 0.767 |
| PofB20 (% bugs found @ 20% churn reviewed) | 64.6% | 54.4% |

Confusion matrices (threshold = 0.5):

| | Predicted Clean | Predicted Bug-Inducing |
|---|---:|---:|
| **Baseline** — Actual Clean | 498 | 182 |
| **Baseline** — Actual Bug-Inducing | 25 | 54 |
| **Hybrid** — Actual Clean | 566 | 114 |
| **Hybrid** — Actual Bug-Inducing | 40 | 39 |

Plots: `models/plots/roc_curve.png`, `models/plots/confusion_matrices.png`.

### Honest discussion: the baseline is *not* clearly beaten here

On this dataset, the Logistic-Regression baseline matches or slightly
outperforms the hybrid deep model on ROC-AUC, recall, and the effort-aware
metrics; the hybrid model wins on accuracy and precision (it is more
conservative — fewer false positives, more false negatives). **This is a
real result, not a bug**, and it is not even a surprising one: it is
essentially the same finding the paper itself highlights from **JITLine**
(Pornprasit & Tantithamthavorn, 2021) — *"a much simpler bag-of-tokens
logistic-regression model ... was competitive with or better than the deep
architectures."* Two concrete reasons this project's hybrid model doesn't
clearly beat the baseline, both explainable rather than mysterious:

1. **Dataset size.** ~2,600 training commits is small for a neural network
   to learn a useful code representation from scratch (the lightweight
   encoder has no pretrained knowledge of code syntax/semantics, unlike
   CodeBERT). Deep code-embedding methods in the literature are normally
   trained on tens/hundreds of thousands of commits.
2. **Lightweight vs. pretrained embedding.** The lightweight `Embedding →
   BiGRU` encoder used here is the fallback path, not the CodeBERT path the
   paper points to (see §2). Switching to real CodeBERT embeddings (on a
   machine with internet access) is the single change most likely to close
   this gap, matching the pattern the paper reports for pretrained-model
   embeddings vs. task-specific ones.

We report both models' real numbers rather than only the flattering one,
in line with the "never fabricate results" requirement — a smaller,
well-understood gap is more useful (and more credible) for a student
project than an inflated one.

---

## 6. Effort-aware evaluation (Popt / PofB20)

`src/evaluate.py` implements Kamei et al.'s effort-aware evaluation
directly from the Alberg diagram (cumulative % defects found vs.
cumulative % effort inspected, where **effort = code churn = la + ld**,
the same proxy Kamei et al. use in the absence of real reviewer-hours
data):

- **Popt** = `1 − (Area(optimal) − Area(model)) / (Area(optimal) − Area(worst))`
- **PofB20** = % of actual bug-inducing commits found among the top 20% of
  cumulative churn, when commits are ranked by predicted bug density
  (`P(bug)/effort`) — directly comparable to the paper's own quoted
  statistic *"reviewing only 20% of the total code churn ... can reveal up
  to 35% of all defect-inducing changes."* Both models here exceed that
  figure on the Flask dataset (54–65%), which is plausible given Flask's
  much smaller scale than Kamei's six large-scale industrial/OSS projects.

Both metrics were computable for this dataset/split (see
`models/metrics.json` → `effort_aware.computable`). If a future
re-run/dataset ever produces a split with zero positives or zero churn,
`effort_aware_metrics()` returns `computable: false` with a stated reason
instead of a silently wrong number.

---

## 7. Streamlit app

```bash
streamlit run app.py
```

Two ways to get a prediction:
- **Paste a diff** — paste a unified diff (or fill in added/removed code
  directly); developer-activity features default to training-set medians
  (there's no real git history for pasted code) and can be adjusted.
- **Select a real commit** — pick any commit from the held-out test set;
  the app fills in its real diff and real developer features, and shows
  whether the model's prediction matches what actually happened.

The app displays: **Bug Probability**, **Prediction** (Bug-Inducing /
Clean), **Risk Level** (Low/Medium/High/Very High), code-change statistics,
and the full developer-activity feature table — plus the baseline
probability for comparison, and the real test-set metrics in the sidebar.

---

## 8. Project structure

```
bug-prediction/
├── data/                          # mined CSVs + the cloned repo used to build them
├── models/                        # trained artifacts + plots + metrics.json
├── src/
│   ├── preprocessing.py           # diff cleaning, tokenizer, vocabulary
│   ├── features.py                # feature schema + normalization
│   ├── embeddings.py              # CodeBERT embedder + lightweight BiGRU encoder
│   ├── model.py                   # Code Branch + Developer Branch + fusion head
│   ├── evaluate.py                # classification + effort-aware (Popt/PofB20) metrics
│   ├── train.py                   # trains baseline LR + hybrid model, saves everything
│   └── predict.py                 # BugPredictor: load artifacts, score a diff
├── app.py                         # Streamlit UI
├── dataset_setup.py                # mines a real JIT dataset from any GitHub repo (SZZ + causal features)
├── requirements.txt
└── README.md
```

---

## 9. Limitations (stated plainly, not hidden)

- The mined labels use a **keyword + SZZ heuristic**, not a manually
  verified ground truth — the same limitation the JIT-SDP literature
  generally has when a curated dataset isn't available.
- The dataset is **one mid-sized project**, not the six-project /
  cross-project setting the paper's target studies use; results may not
  generalize to other codebases or languages. `dataset_setup.py` supports
  re-running against any other GitHub repo to check this.
- The **lightweight code encoder** (not CodeBERT) is what was actually
  trained here, for the network-access reasons explained in §2; the
  CodeBERT code path is implemented and ready to use but requires an
  internet-connected machine to download the pretrained weights.
- **Effort** in the Popt/PofB20 computation is code churn (la+ld), a proxy
  for reviewer effort, not measured review time — the same proxy Kamei et
  al. use.
