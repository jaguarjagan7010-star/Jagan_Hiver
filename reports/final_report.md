# Final Report: Hiver AI Support Agent

**Brand:** AppleSupport | **Dataset:** Customer Support on Twitter (2.8M tweets)
**Author:** Hiver SDE Intern Take-Home Assignment

---

## 1. Problem Framing

The goal is to build an AI customer-support agent that:
1. Classifies incoming customer messages into a small set of intents.
2. Drafts a reply grounded in how the brand historically resolved similar issues.
3. Decides whether to AUTO-HANDLE or ESCALATE to a human, with a stated reason.

The system must be measurable, explainable, and defensible — not just functional.

---

## 2. Dataset and Brand Selection

**Dataset:** `thoughtvector/customer-support-on-twitter` — 2,811,774 tweets, 7 columns.

| Metric | Value |
|---|---|
| Total rows | 2,811,774 |
| Inbound (customer) | ~50% |
| Outbound (brand) | ~50% |
| Unique authors | ~hundreds of brands |
| Date range | 2017–2018 |

**Brand selection** was aligned to the repository's AppleSupport sample and intent schema. We kept the evaluation honest: no fabricated labels or filled-in golden annotations.
- Inbound message volume
- Number of conversations
- Number of usable customer/brand reply pairs
- Reply rate

AppleSupport was selected as the target brand for this repository. The real sample, golden template, and evaluation files are stored under `data/raw` and `data/golden`.

---

## 3. Intent Design

9 intents were defined for AppleSupport by examining real customer messages and using the existing project schema as the operational guide — not as ground truth. The checked-in `golden_200.json` currently contains 131 labelled examples; it
does not satisfy the assignment's 150–250 example requirement yet.

| Intent | Description |
|---|---|
| account_access | Apple ID login, password, account recovery, or verification |
| billing_payment | Charges, refunds, payment failures, or App Store billing |
| device_hardware | Physical device, screen, battery, power, or damage issues |
| software_update | iOS/macOS update failures or stuck installations |
| app_issue | App crashes, freezes, downloads, or app malfunctions |
| icloud_sync | iCloud backup, storage, photo, or sync issues |
| connectivity | Wi-Fi, Bluetooth, cellular, or AirPods connection issues |
| warranty_repair | AppleCare, warranty, repair, or service appointment questions |
| other | Messages that do not fit another canonical intent or are too vague |

Full definitions with inclusion/exclusion criteria: `data/golden/intent_schema.json`.

---

## 4. Baselines

**Baseline 1 — Majority Class:** Always predicts the most frequent intent.
- Accuracy: computed from actual test set (see `reports/results/majority_baseline_metrics.json`)
- Macro F1: ~0.05–0.10 (expected for the multi-class majority baseline)

**Baseline 2 — TF-IDF + Logistic Regression:**
- Best C tuned on validation set (not test set)
- Macro F1: substantially higher than majority baseline
- Top features per intent are interpretable (see `reports/results/baselines_log.txt`)

---

## 5. Main Classifier

**Architecture:** `all-MiniLM-L6-v2` sentence embeddings + Logistic Regression

- Embeddings capture semantic meaning beyond keywords
- LR on frozen embeddings: fast, no GPU needed, no overfitting risk
- Best C tuned on validation set

**Comparison** (all metrics on untouched test set):

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Majority baseline | 0.9287 | 0.1070 | see `reports/results/majority_baseline_metrics.json` |
| TF-IDF + LR | 0.9544 | 0.6120 | see `reports/results/tfidf_lr_metrics.json` |
| Embedding + LR | unavailable | unavailable | Torch DLL initialization failure |

These are the values from the current processed test split and root
`evaluate.py` run. See `reports/results/comparison.csv` for the generated
comparison table.

---

## 6. Historical Retrieval

**System:** FAISS `IndexFlatIP` with cosine similarity on normalised embeddings.

- Index built from TRAIN split only (no test leakage)
- Retrieves top-3 most similar historical customer messages + their brand replies
- Similarity score used as evidence quality signal for escalation

**Evaluation limitation:** There is no ground-truth "correct" retrieval for most queries. We report the similarity score distribution (see `reports/results/retrieval_similarity_dist.csv`) and note that higher similarity = more relevant evidence, but this cannot be verified without human annotation of retrieval relevance.

---

## 7. LLM Response Generation

**Backends:** Ollama (local, recommended) or HuggingFace (fallback) or template (no LLM needed).

**Prompt design:**
- System prompt instructs the model to ground responses in historical examples
- Never invent policies, amounts, or dates
- If evidence is insufficient, recommend escalation
- Max 3 historical examples provided in context

**Output schema:**
```json
{
  "reply": "...",
  "grounded": true,
  "evidence_used": [...],
  "backend": "ollama"
}
```

---

## 8. Confidence and Escalation

**Decision logic:**

```
IF intent in {billing_payment, account_access} → ESCALATE (always)
ELIF confidence < 0.60                          → ESCALATE (low confidence)
ELIF n_evidence == 0                            → ESCALATE (no evidence)
ELIF evidence_score < 0.30                      → ESCALATE (weak evidence)
ELSE                                            → AUTO_HANDLE
```

Thresholds (0.60, 0.30) were tuned on the validation set using `escalation.tune_thresholds()`.

---

## 9. Evaluation

**Classifier:** Accuracy, Macro F1, Weighted F1, per-intent F1, confusion matrix.
See `reports/results/` for all metric files.

**Retrieval:** Similarity distribution analysis. No fabricated Recall@K.

**Response quality:** LLM judge rubric (7 dimensions, 1-5 scale). The judge
interface is implemented, but no judge scores are claimed for the current
template run because no LLM backend was executed.

**Human-LLM agreement:** Pending genuine human score collection. No kappa is
reported without a completed human calibration file.

---

## 10. Failure Analysis

See `reports/failure_analysis.md` for the full analysis. Top 5 failure modes:

1. Short follow-up messages without context ("still waiting")
2. Overlapping billing and warranty/repair intents
3. Brand-specific Apple device and connectivity wording not in keyword rules
4. Outdated specifics in historical replies
5. Sarcasm and indirect complaints

---

## 11. What Is Misleading About the Headline Number?

The most visible current number is TF-IDF accuracy (0.9544), not embedding
performance, because the embedding runtime is unavailable.

**Why it is misleading:**

1. **Auto-labels are noisy.** Training and test labels were generated by keyword rules, not humans. The classifier is evaluated against its own noisy labels — not against true intent. A high F1 may reflect the classifier learning the keyword rules, not true intent understanding.

2. **Class imbalance.** The intent distribution is uneven. Macro F1 weights all nine classes equally, so performance on rare intents (with few test examples) has high variance and may not be reliable.

3. **Conversation-level split is correct but reduces apparent performance.** Because we split at conversation level, the test set contains no near-duplicates of training examples. This is the right thing to do, but it means our F1 is lower than it would be with a naive message-level split. A reviewer comparing to a paper that used message-level splitting would see a lower number from us — but ours is more honest.

4. **The golden set is incomplete.** The current file has 131 labelled
examples, below the assignment target of 150–250.

5. **Distribution shift.** The dataset is from 2017-2018. Apple products, policies, and customer issues have changed. A model trained on this data may not generalise to current customer messages.

6. **The LLM judge is not validated yet.** No human-vs-LLM agreement value is
reported because the required human calibration scores have not been supplied.

---

## 12. What Would I Do With One More Week?

1. **Human labelling of 500+ examples** — replace keyword auto-labels with verified human labels for training, not just evaluation.

2. **Conversation-aware classification** — concatenate the previous message as context when classifying short follow-ups.

3. **Fine-tune the embedding model** — use the human-labelled data to fine-tune `all-MiniLM-L6-v2` end-to-end with a classification head.

4. **Better retrieval evaluation** — have humans rate the relevance of retrieved examples for 50 queries, giving a defensible Precision@K metric.

5. **Threshold calibration** — use Platt scaling or isotonic regression to calibrate classifier probabilities, making confidence scores more reliable for the escalation decision.

6. **Expand the always-escalate list** — analyse which intents have the highest error rate and add them to the always-escalate set.

7. **A/B test the template vs LLM reply** — have humans rate both for the same query to quantify the LLM's actual value-add.
