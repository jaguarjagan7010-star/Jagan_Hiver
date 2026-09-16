# Hiver AI Support Agent

An AppleSupport customer-support agent built on the
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset.

Given a customer message, the system:
1. **Classifies** the message into one of 9 AppleSupport intents
2. **Retrieves** the most similar historical support interactions
3. **Drafts** a reply grounded in real historical responses
4. **Decides** AUTO-HANDLE or ESCALATE with a stated reason

> Important: the repository keeps the human-labelling fields intentionally blank until a human annotator fills them. No fabricated labels or fake evaluation numbers are included.

---

## Architecture

```
Customer message
      |
      v
  [Cleaner]  — remove @mentions, URLs, whitespace
      |
      v
  [Classifier]  — sentence embedding + logistic regression
      |                → predicted intent + confidence
      v
  [Retriever]  — FAISS cosine similarity search
      |                → top-3 historical customer/brand pairs
      v
  [Generator]  — LLM (Ollama/HuggingFace/template fallback)
      |                → draft reply grounded in evidence
      v
  [Escalation]  — rule-based decision layer
                       → AUTO_HANDLE or ESCALATE + reason
```

---

## Dataset

- **Source:** [thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
- **Size:** 2,811,774 tweets
- **Brand used:** AppleSupport (selected objectively — see `reports/decision_log.md`)
- **Date range:** 2017–2018

---

## Setup

### Requirements
- Python 3.10 (the supplied environment uses Python 3.10.11)
- ~2 GB disk space for models
- No GPU required (CPU-only)

### Installation

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd hiver-support-agent

# 2. Create/use the supported environment
py -3.10 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Copy environment template
copy .env.example .env
# Edit .env to set LLM_BACKEND (ollama recommended)
```

### Data Download

```bash
# Option A: Kaggle API (add credentials to .env first)
python scripts/download_data.py

# Option B: Manual
# 1. Go to https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
# 2. Download and unzip
# 3. Copy twcs.csv into data/raw/twcs/twcs.csv
```

---

## Running the Pipeline

### Quick demo (under 15 minutes on a laptop)

```bash
# Step 1: Prepare data + train baselines (~5 min)
python scripts/train_baselines.py

# Step 2: Train embedding classifier (~5 min, downloads ~80MB model)
python scripts/train_classifier.py

# Step 3: Build retrieval index (~3 min)
python scripts/build_retrieval.py

# Step 4: Run demo
python scripts/run_pipeline.py --demo-only --backend template
```

### Interactive CLI

```bash
python -m src.pipeline
```

Example session:
```
Customer message: My iPhone won't connect to Wi-Fi

============================================================
Brand:      AppleSupport
Intent:     connectivity  (confidence: 0.87)

Historical evidence (3 examples):
  [1] sim=0.821
      Customer: My iPhone won't connect to Wi-Fi...
      Support:  We're sorry you're having connectivity issues. Please DM us...
  ...

Draft reply:
  We're sorry you're having connectivity issues. Please DM us and we'll
  troubleshoot together.

Decision:   AUTO_HANDLE
Reason:     High classifier confidence (0.87) and strong historical
            evidence (score=0.82, 3 examples retrieved).
============================================================
```

### Single message

```bash
python scripts/run_pipeline.py --message "I can't log into my account" --backend template
```

---

## Training

```bash
# Baselines (majority + TF-IDF+LR)
python scripts/train_baselines.py

# Main classifier (sentence embeddings + LR)
python scripts/train_classifier.py

# Retrieval index
python scripts/build_retrieval.py
```

---

## Evaluation

```bash
python evaluate.py
```

The root `evaluate.py` is the canonical evaluation entry point. It evaluates
the current test split and available classifier artifacts, reads the
human-labelled `data/golden/golden_200.json` when no explicit golden set is
provided, and writes results to `reports/results/`:
- `comparison.csv` — model comparison table
- `*_metrics.json` — per-model metrics
- `*_confusion_matrix.csv` — confusion matrices
- `evaluation_log.txt` — full evaluation log

Use an explicit golden set or LLM judge backend when needed:

```bash
python evaluate.py --golden_set golden_200.json
python evaluate.py --golden_set golden_200.json --backend ollama
```

---

## Results

> Run the pipeline to generate actual numbers. All results are computed from real data.
> See `reports/results/comparison.csv` after running training scripts.

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Majority baseline | 0.9287 | 0.107 | generated in `reports/results/` |
| TF-IDF + LR | 0.9544 | 0.612 | generated in `reports/results/` |
| Embedding + LR | unavailable | unavailable | Torch DLL failure in current environment |

These are the results from the checked-in processed test split and the current
evaluation run; rerunning evaluation may replace the generated result files.

### Golden-set status

`data/golden/golden_200.json` currently contains **131 genuinely labelled
examples**, not 200. The labels are not inferred by the evaluator. To meet a
150–250 example assignment requirement, create or extend an annotation queue
from the real test split, then label the additional rows manually:

```powershell
python -m src.evaluation.golden_set
python -m src.evaluation.annotation --source data/golden/golden_set.csv
python -m src.evaluation.annotation `
  --source data/golden/golden_200.json `
  --validate `
  --require-complete
```

The annotation command validates intent names against
`data/golden/intent_schema.json`, requires source identifiers and text, and
rejects duplicate IDs or tweet IDs. Do not copy `auto_intent` into
`gold_intent`; read each message and label it manually.

### LLM judge and human calibration

The LLM judge interface and seven-dimension 1–5 rubric are implemented in
`src/evaluation/llm_judge.py`. It runs only when `--backend ollama` or
`--backend huggingface` is selected. No judge scores are claimed for the
current `template` run. Human calibration is also pending: fill a generated
human score file with the same IDs and rubric columns before running
`src/evaluation/human_agreement.py`. Cohen's kappa, exact agreement, Pearson
correlation, and MAE must only be reported after real human scores exist.

Create the blank calibration template (50 real examples, no fabricated scores):

```powershell
python -m src.evaluation.human_agreement `
  --source data/golden/golden_set.csv `
  --output data/golden/human_scores_50.csv `
  --n 50
```

The generated file contains blank `generated_reply` and seven blank 1–5 rubric
columns. It is a manual calibration template, not evidence that judging has
already been completed. Populate it only after running the judge on the same
50 examples.

---

## Failure Analysis

See `reports/failure_analysis.md`. Top 5 failure modes:

1. Short follow-ups without context ("still waiting")
2. Overlapping billing and warranty/repair intents
3. Brand-specific jargon not in keyword rules
4. Outdated specifics in historical replies
5. Sarcasm and indirect complaints

---

## Design Decisions

15 non-obvious decisions documented in `reports/decision_log.md`. Key ones:

- **Why AppleSupport?** Appropriate Apple-specific support issues and a real AppleSupport sample set in the repository
- **Why 9 intents?** The canonical schema preserves distinct AppleSupport resolution paths
- **Why conversation-level split?** Prevents near-duplicate leakage across train/test
- **Why all-MiniLM-L6-v2?** Best quality/speed tradeoff for CPU (~80MB, fast)
- **Why always escalate billing/account?** Cost of wrong auto-response is too high

---

## Limitations

1. Auto-labels (keyword rules) are noisy — not human-verified ground truth
2. Dataset is from 2017-2018 — distribution shift to current issues
3. LLM judge validated on only ~30 examples — not statistically reliable at scale
4. No conversation context in classifier — short follow-ups are ambiguous
5. The current Windows environment cannot import Torch (`WinError 1114` while
   loading `torch\lib\c10.dll`), so embedding and FAISS-dependent evaluations
   are reported as unavailable until the environment is repaired.
6. Template fallback returns historical replies verbatim — may be outdated

## What is misleading about the headline number?

The TF-IDF accuracy of 0.9544 is not a measure of broad support quality.
The test labels are keyword-derived auto-labels, the class distribution is
imbalanced, and the majority baseline already reaches 0.9287 accuracy while
achieving only 0.107 macro F1. Macro F1 is therefore the more informative
classifier comparison, but it is still measured against noisy labels rather
than a fully human-labelled test set. The 131-row golden set is below the
assignment target and the embedding/retrieval path is unavailable in the
current Torch environment.

## What I'd do next with one more week

1. Manually extend the golden set to 150–250 rows and double-label 50 rows.
2. Repair the Windows Torch installation and rerun embedding/retrieval tests.
3. Collect human reply-quality scores and compute judge agreement.
4. Add conversation context for short follow-up messages.
5. Calibrate escalation thresholds on the validation split.
6. Add human relevance labels for retrieval Precision@K.
7. Retrain on a larger, human-verified intent sample and monitor drift.

See `reports/final_report.md` Section 11 for full discussion.

---

## Reproduction

```bash
# Full reproducible run from scratch:
pip install -r requirements.txt
# (place twcs.csv in data/raw/twcs/twcs.csv)
python scripts/train_baselines.py
python scripts/train_classifier.py
python scripts/build_retrieval.py
python scripts/run_pipeline.py --demo-only --backend template
python evaluate.py
```

All random seeds are fixed at 42. Results are deterministic.

---

## Project Structure

```
hiver-support-agent/
├── src/
│   ├── config.py              # All settings in one place
│   ├── data/                  # Loader, cleaner, conversations, sampling
│   ├── brand/                 # Brand selection
│   ├── intents/               # Discovery, labeling, classifiers
│   ├── retrieval/             # FAISS index + retriever
│   ├── generation/            # Prompts + LLM generator
│   ├── decision/              # Escalation logic
│   ├── evaluation/            # Metrics, golden set, LLM judge, agreement
│   └── pipeline.py            # End-to-end pipeline
├── scripts/                   # One script per phase
├── data/
│   ├── raw/                   # Original dataset (gitignored)
│   ├── processed/             # Train/val/test splits, models, FAISS index
│   └── golden/                # Intent schema + golden evaluation set
├── reports/
│   ├── decision_log.md        # 15 non-obvious decisions
│   ├── failure_analysis.md    # Top 5 failure modes
│   ├── final_report.md        # Full report
│   ├── figures/               # Charts
│   └── results/               # Metrics, comparison tables
├── evaluate.py               # canonical evaluation entry point
├── run_pipeline.py           # training/demo pipeline runner
├── tests/                     # pytest unit tests
└── outputs/                   # Predictions, evaluation outputs, logs
```

---

## Citation

```
Dataset:
  @misc{thoughtvector2017twitter,
    title  = {Customer Support on Twitter},
    author = {thoughtvector},
    year   = {2017},
    url    = {https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter}
  }

Embedding model:
  Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using
  Siamese BERT-Networks. EMNLP 2019. https://www.sbert.net

Vector search:
  Johnson, J., Douze, M., & Jegou, H. (2019). Billion-scale similarity search
  with GPUs. IEEE Transactions on Big Data. https://github.com/facebookresearch/faiss

Libraries: pandas, numpy, scikit-learn, sentence-transformers, faiss-cpu,
           matplotlib, seaborn, pytorch, pytest, python-dotenv, requests, tqdm
```
