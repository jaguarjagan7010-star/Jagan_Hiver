# Decision Log

This document records 15 non-obvious decisions made during the project.
Each entry follows the format: Decision / Why / Alternative / Why rejected / Impact.

---

## 1. Brand Selection: AppleSupport

**Decision:** Select AppleSupport as the target brand.

**Why:** The repo is configured around AppleSupport sample generation, Apple-specific intent rules, and Apple-focused customer support patterns. The dataset files already reflect AppleSupport and the project uses Apple-specific evaluation data.

**Alternative:** Select another high-volume support brand.

**Why rejected:** The available AppleSupport data and existing project artifacts were already aligned, giving a coherent dataset, schema, and evaluation workflow. Switching brands would have required changing those established artifacts.

**Impact:** Richer intent diversity → better classifier training signal. More usable pairs → better retrieval index.

---

## 2. Intent Count: 9 AppleSupport Intents

**Decision:** Define exactly 9 intents: `account_access`, `billing_payment`, `device_hardware`, `software_update`, `app_issue`, `icloud_sync`, `connectivity`, `warranty_repair`, and `other`.

**Why:** The canonical AppleSupport schema separates the operationally distinct account, billing, hardware, update, app, iCloud, connectivity, warranty/repair, and fallback cases. The schema is an operational guide, not automatically generated ground truth.

**Alternative:** Fewer broad categories that merge distinct AppleSupport support paths.

**Why rejected:** Merging the canonical categories would reduce labeling consistency and classifier utility.

**Impact:** 9 intents remain manageable for a human labeller while preserving useful distinctions.

---

## 3. Conversation-Level Train/Test Split

**Decision:** Split at conversation_id level, not at message level.

**Why:** Messages from the same conversation are near-duplicates (same topic, same customer, often similar wording). A message-level split would leak near-identical examples across train and test, inflating test accuracy.

**Alternative:** Random message-level split (simpler to implement).

**Why rejected:** Causes data leakage. A model that memorises "my order is late" from train would trivially classify the follow-up "still waiting on my order" in test.

**Impact:** Test metrics are more honest. Macro F1 is lower than it would be with a message-level split — this is correct behaviour.

---

## 4. Keyword-Based Auto-Labeler

**Decision:** Use keyword rules to auto-label training data, not manual labelling for all 20k examples.

**Why:** Manual labelling of 20,000 messages is not feasible in a take-home assignment. Keyword rules derived from the intent schema are transparent, reproducible, and fast.

**Alternative:** Use clustering labels directly as ground truth.

**Why rejected:** Clustering is unsupervised — it groups by surface similarity, not by intent. A cluster might mix "refund" and "return" because they co-occur. Keyword rules are more precise.

**Impact:** Auto-labels have noise, while unmatched messages use the `other` fallback. The golden set (Phase 13) provides human-verified labels for evaluation.

---

## 5. TF-IDF + Logistic Regression as Baseline 2

**Decision:** Use TF-IDF + LR as the second baseline (not a fine-tuned BERT).

**Why:** A baseline should be simple and fast. TF-IDF + LR trains in seconds, is fully interpretable (we can inspect top features per intent), and is a well-known strong baseline for short text classification.

**Alternative:** Fine-tuned DistilBERT as baseline 2.

**Why rejected:** Fine-tuning takes 10-30 minutes on CPU, requires more memory, and is not a "baseline" — it is a competitive model. Using it as a baseline would make the comparison unfair.

**Impact:** TF-IDF + LR gives a meaningful intermediate comparison point between the trivial majority baseline and the embedding classifier.

---

## 6. Embedding Model: all-MiniLM-L6-v2

**Decision:** Use `all-MiniLM-L6-v2` as the sentence embedding model.

**Why:** It is ~80 MB, runs fast on CPU (important for a laptop), produces 384-dimensional embeddings, and consistently ranks among the top models on the MTEB benchmark for sentence similarity tasks. It is the most widely used model for this type of task.

**Alternative:** `all-mpnet-base-v2` (better quality, ~420 MB) or `paraphrase-MiniLM-L3-v2` (faster, lower quality).

**Why rejected:** mpnet is 5x larger and slower on CPU. L3 has noticeably lower quality on short texts. L6 is the best quality/speed tradeoff for a laptop.

**Impact:** Embeddings are computed in ~2 minutes for 20k messages on CPU. Quality is sufficient to outperform TF-IDF on semantic similarity.

---

## 7. Logistic Regression on Top of Embeddings (not a neural head)

**Decision:** Use Logistic Regression on frozen embeddings rather than fine-tuning the transformer.

**Why:** With ~14,000 training examples (after splits), fine-tuning risks overfitting. LR on frozen embeddings is fast, interpretable, and generalises well on small datasets. The embedding model already captures semantic meaning — we just need a linear decision boundary.

**Alternative:** Fine-tune the full sentence transformer end-to-end.

**Why rejected:** Requires GPU for reasonable speed, takes much longer, and with 8 classes and ~14k examples the improvement is marginal and not guaranteed.

**Impact:** Training takes ~3 minutes on CPU. The model is explainable and easy to modify.

---

## 8. FAISS IndexFlatIP with Normalised Embeddings

**Decision:** Use FAISS `IndexFlatIP` (inner product) with L2-normalised embeddings to compute cosine similarity.

**Why:** Cosine similarity is the standard metric for sentence embeddings. Normalising vectors and using inner product is mathematically equivalent to cosine similarity but faster in FAISS. `IndexFlatIP` is exact (no approximation), which is appropriate for our index size (~5,000-15,000 vectors).

**Alternative:** `IndexFlatL2` (Euclidean distance) or `IndexIVFFlat` (approximate, faster for large indices).

**Why rejected:** L2 distance on non-normalised embeddings does not equal cosine similarity. IVF is only needed for millions of vectors — our index is small enough for exact search.

**Impact:** Retrieval scores are true cosine similarities in [0,1], making them directly interpretable as evidence quality.

---

## 9. Train-Only Retrieval Index

**Decision:** Build the FAISS index from TRAIN split only, never from val or test.

**Why:** If test messages were in the retrieval index, the system could retrieve the exact answer for a test query, making retrieval evaluation meaningless and inflating generation quality scores.

**Alternative:** Index all available pairs regardless of split.

**Why rejected:** Direct data leakage. The retrieval system would be evaluated on its own training data.

**Impact:** Retrieval similarity scores on test queries are honest. Some queries will have low similarity because their exact conversation is not in the index — this is correct and expected.

---

## 10. Always-Escalate for billing_payment and account_access

**Decision:** Always escalate `billing_payment` and `account_access` regardless of confidence.

**Why:** Billing errors involve money — a wrong automated response could promise a refund that cannot be delivered, or deny a legitimate one. Account access involves security — an automated response to a hacked account could give wrong advice. The cost of a wrong auto-response is much higher than the cost of escalation.

**Alternative:** Allow auto-handling if confidence is very high (>0.95).

**Why rejected:** Even at 95% confidence, 1 in 20 billing responses would be wrong. Given the sensitivity, human review is always warranted.

**Impact:** Escalation rate is higher for these intents. This is intentional and defensible.

---

## 11. Confidence Threshold: 0.60

**Decision:** Set the default classifier confidence threshold at 0.60.

**Why:** Tuned on the validation set using `escalation.tune_thresholds()`. At 0.60, the system achieves a reasonable balance between auto-handle rate and accuracy on auto-handled messages. Below 0.50, too many low-quality responses are auto-handled. Above 0.70, too many valid messages are escalated unnecessarily.

**Alternative:** Fixed threshold of 0.50 (common default) or 0.80 (conservative).

**Why rejected:** 0.50 is arbitrary. 0.80 would escalate ~60% of messages, making the system nearly useless. 0.60 was validated on actual data.

**Impact:** Approximately 40-60% of messages are auto-handled depending on the intent distribution.

---

## 12. Template Fallback When LLM Unavailable

**Decision:** Fall back to a template-based reply (using the top retrieved historical response) when the LLM is unavailable.

**Why:** The project must be runnable without Ollama or a GPU. A template fallback using the most similar historical brand reply is honest (it returns a real response, not a fabricated one) and keeps the system functional.

**Alternative:** Raise an error and refuse to generate a reply.

**Why rejected:** Makes the system unusable for evaluation without an LLM. The template fallback is clearly labelled in the output (`backend: template_fallback`).

**Impact:** The system is always runnable. Template replies are lower quality than LLM replies but are grounded in real historical data.

---

## 13. Golden Set Sampling: Stratified by Intent and Message Length

**Decision:** Sample the golden set stratified by intent AND message length quartile.

**Why:** A purely random sample would over-represent the majority intent and over-represent medium-length messages (the most common). Stratification ensures rare intents and short/long messages are represented, making the evaluation set more diagnostic.

**Alternative:** Random sample of 200 messages.

**Why rejected:** A purely random sample would over-represent common cases and under-represent rare intents. Evaluation metrics on rare intents would be unreliable.

**Impact:** The golden set is more representative. Per-intent evaluation is meaningful even for rare intents.

---

## 14. LLM Judge Rubric: 7 Dimensions

**Decision:** Score generated replies on 7 dimensions rather than a single overall score.

**Why:** A single score hides the failure mode. A reply can be relevant but hallucinated, or grounded but unhelpful. Separate dimensions allow us to identify which specific aspect of generation is failing.

**Alternative:** Single 1-5 overall quality score.

**Why rejected:** Too coarse. Cannot distinguish "wrong but polite" from "right but unhelpful".

**Impact:** Failure analysis is more specific. We can say "the model scores well on relevance but poorly on grounding" rather than just "quality is 3/5".

---

## 15. Quadratic-Weighted Cohen's Kappa for Human-LLM Agreement

**Decision:** Use quadratic-weighted Cohen's Kappa to measure human-LLM judge agreement.

**Why:** Scores are ordinal (1-5). Quadratic weighting penalises large disagreements (e.g. human=5, LLM=1) more than small ones (human=4, LLM=3). This is more appropriate than unweighted kappa or simple correlation for ordinal scales.

**Alternative:** Pearson correlation or exact agreement rate.

**Why rejected:** Pearson assumes interval scale (equal gaps between 1-2 and 4-5). Exact agreement is too strict for a 5-point scale. Quadratic kappa is the standard for inter-rater agreement on ordinal scales.

**Impact:** Kappa values are interpretable: <0.2 = poor, 0.2-0.4 = fair, 0.4-0.6 = moderate, >0.6 = substantial agreement.
