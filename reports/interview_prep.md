# Interview Preparation

Realistic Hiver interview questions based on this actual implementation.
For each question: interviewer intent, strong answer, where the evidence lives.

---

## Q1. Why did you choose AppleSupport?

**Interviewer intent:** Did you make a data-driven decision or just pick a famous brand?

**Strong answer:**
"I aligned the project to AppleSupport because the repository's sample and golden files are specifically AppleSupport-oriented. The goal is not to fabricate labels or rankings; instead, the system uses real AppleSupport sample data, Apple-specific intent rules, and a human-label template that is intentionally left blank until an annotator fills it in. This keeps the evaluation honest and avoids claiming unsupported labels."

**Evidence:** `reports/results/brand_ranking.csv`, `reports/decision_log.md` Decision #1.

---

## Q2. Why did you choose these 9 intents?

**Interviewer intent:** Did you invent intents or derive them from data?

**Strong answer:**
"The intent schema is AppleSupport-specific and is used as an operational guide. The repo keeps the gold labels as empty placeholders because the project is not allowed to fabricate human annotations. That is a deliberate safeguard: the final evaluation should use real human labels, not guessed values."

"The nine canonical intents are account_access, billing_payment, device_hardware, software_update, app_issue, icloud_sync, connectivity, warranty_repair, and other. They preserve distinct AppleSupport resolution paths while keeping annotation manageable."

**Evidence:** `scripts/build_intents.py`, `data/golden/intent_schema.json`, `reports/results/cluster_discovery.csv`.

---

## Q3. How did you label the training data?

**Interviewer intent:** Do you understand the difference between auto-labels and human labels?

**Strong answer:**
"I used keyword rules derived from the intent schema to auto-label the training data. The rules are in `src/intents/labeling.py`. Messages that match no rule use the `other` fallback. I'm explicit that these are auto-labels, not human labels. The golden evaluation template is the only place where human-verified labels are used — and its gold columns remain empty until a human fills them in."

**Evidence:** `src/intents/labeling.py`, `data/golden/golden_set.csv`.

---

## Q4. Why did you use conversation-level splitting?

**Interviewer intent:** Do you understand data leakage?

**Strong answer:**
"Messages from the same conversation are near-duplicates — same customer, same issue, often similar wording. If I split at message level, a message like 'my order is late' in training and 'still waiting on my order' in test would be treated as independent, but they're from the same conversation. The classifier would effectively memorise the training example and get the test example right for the wrong reason. Conversation-level splitting prevents this. I also added an assertion in `split_conversations()` that raises an error if any conversation_id appears in both train and test."

**Evidence:** `src/intents/labeling.py` → `split_conversations()`, `reports/data_leakage_audit.md`.

---

## Q5. Why TF-IDF as a baseline?

**Interviewer intent:** Can you justify your architectural choices?

**Strong answer:**
"A baseline should be simple, fast, and interpretable. TF-IDF + Logistic Regression trains in seconds on CPU, requires no model download, and is fully interpretable — I can call `get_top_features(intent)` to see exactly which words drive each prediction. It's also a well-known strong baseline for short text classification. Using a fine-tuned BERT as a 'baseline' would be misleading — that's a competitive model, not a baseline."

**Evidence:** `src/intents/classifier.py` → `TfidfLRClassifier`, `reports/decision_log.md` Decision #5.

---

## Q6. Why did you choose your main classifier architecture?

**Interviewer intent:** Do you understand the tradeoffs?

**Strong answer:**
"Sentence embeddings from `all-MiniLM-L6-v2` capture semantic meaning that TF-IDF misses — for example, 'my iPhone will not power on' and 'the phone refuses to start' are semantically similar despite different wording. I put Logistic Regression on top of frozen embeddings rather than fine-tuning the full transformer because: (1) with ~14,000 training examples, fine-tuning risks overfitting; (2) LR is fast and interpretable; (3) the embedding model already captures the semantic structure we need. The model is ~80MB and runs in ~3 minutes on CPU."

**Evidence:** `src/intents/classifier.py` → `EmbeddingClassifier`, `reports/decision_log.md` Decision #6, #7.

---

## Q7. Where exactly is confidence calculated?

**Interviewer intent:** Can you point to the specific code?

**Strong answer:**
"In `src/intents/classifier.py`, the `predict_with_confidence()` method calls `predict_proba()` which returns a dict of `{intent: probability}` for each class. Confidence is the maximum probability across all classes — `confidence = max(proba_dict.values())`. For Logistic Regression, these probabilities come from the softmax of the decision function. For the majority baseline, confidence is always 1.0 because it always predicts the same class with certainty."

**Evidence:** `src/intents/classifier.py` → `predict_with_confidence()`.

---

## Q8. How did you choose the confidence threshold?

**Interviewer intent:** Did you tune it properly or pick it arbitrarily?

**Strong answer:**
"I tuned it on the validation set using `escalation.tune_thresholds()` in `src/decision/escalation.py`. The function loops over candidate thresholds (0.3 to 0.9 in steps of 0.1) and for each threshold computes the macro F1 on the auto-handled validation messages. The threshold that maximises F1 while keeping the auto-handle rate reasonable is selected. The default of 0.60 came from this process. I never touched the test set during threshold selection."

**Evidence:** `src/decision/escalation.py` → `tune_thresholds()`, `reports/decision_log.md` Decision #11.

---

## Q9. What happens if retrieval returns nothing?

**Interviewer intent:** Is your system robust to edge cases?

**Strong answer:**
"Three things happen: (1) The `Retriever.retrieve()` method returns an empty list — it never crashes. (2) The `get_evidence_score()` method returns 0.0 for an empty list. (3) In `escalation.decide()`, `n_evidence == 0` triggers an ESCALATE decision with the reason 'No historical support examples found for this query.' The generator also handles empty evidence gracefully — it falls back to a template reply. This is tested in `tests/test_retrieval.py` → `test_retriever_empty_query()`."

**Evidence:** `src/retrieval/retriever.py`, `src/decision/escalation.py`, `tests/test_retrieval.py`.

---

## Q10. How do you prevent hallucinations?

**Interviewer intent:** Do you understand LLM reliability issues?

**Strong answer:**
"Three layers: (1) The system prompt explicitly instructs the LLM to base replies only on the provided historical examples and never invent policies, amounts, or dates. (2) The escalation layer escalates when evidence is weak — so the LLM only generates replies when there is strong historical grounding. (3) The template fallback returns actual historical brand replies verbatim when the LLM is unavailable. None of these fully prevent hallucination — the LLM judge's `no_hallucination` dimension is specifically designed to detect it, and I report this as a known limitation."

**Evidence:** `src/generation/prompts.py`, `src/generation/generator.py`, `reports/failure_analysis.md` Failure #4.

---

## Q11. How did you evaluate response quality?

**Interviewer intent:** Can you defend your evaluation methodology?

**Strong answer:**
"I used an LLM-as-a-judge rubric with 7 dimensions: correctness, relevance, grounding, helpfulness, brand tone, no-hallucination, and escalation appropriateness. Each is scored 1-5. The judge prompt is in `src/generation/prompts.py` → `build_judge_prompt()`. I also created a human evaluation template (`src/evaluation/human_agreement.py` → `create_human_eval_template()`) so a human can score the same rubric on the same examples. I then compute quadratic-weighted Cohen's Kappa between human and LLM scores. I explicitly state that with ~30 examples, the agreement estimate is not statistically reliable."

**Evidence:** `src/evaluation/llm_judge.py`, `src/evaluation/human_agreement.py`, `reports/final_report.md` Section 9.

---

## Q12. How could your F1 score be misleading?

**Interviewer intent:** Do you understand the limitations of your own evaluation?

**Strong answer:**
"Three main reasons: First, the labels are auto-generated by keyword rules — the classifier is evaluated against its own noisy labels, not true human intent. A high F1 may mean the classifier learned the keyword rules, not true intent. Second, class imbalance — the nine intents are not equally represented. Macro F1 weights all classes equally, so the score is sensitive to performance on rare intents with few test examples. Third, conversation-level splitting is correct but makes our F1 lower than a naive message-level split — a reviewer comparing to a paper using message-level splitting would see a lower number from us, but ours is more honest."

**Evidence:** `reports/final_report.md` Section 11, `reports/decision_log.md` Decision #4.

---

## Q13. How did you prevent data leakage?

**Interviewer intent:** Do you understand all the ways leakage can occur?

**Strong answer:**
"I documented 8 specific leakage checks in `reports/data_leakage_audit.md`. The key ones: (1) Conversation-level split with an assertion that raises an error if any conversation appears in both train and test. (2) The FAISS retrieval index is built from train data only — test conversations are explicitly filtered out in `build_retrieval.py`. (3) The golden set is sampled from the test split only. (4) Threshold tuning uses the validation set, never the test set. (5) C parameter tuning uses validation macro F1, never test."

**Evidence:** `reports/data_leakage_audit.md`, `src/intents/labeling.py`, `scripts/build_retrieval.py`.

---

## Q14. Show me the retrieval code.

**Interviewer intent:** Can you walk through actual code?

**Strong answer:**
"The retriever is in `src/retrieval/retriever.py`. The `retrieve()` method: (1) Lazy-loads the FAISS index and sentence transformer encoder. (2) Embeds the query with L2 normalisation. (3) Calls `self._index.search(embedding, k)` which returns cosine similarity scores and indices. (4) Maps indices back to metadata (customer_text, brand_text, conversation_id). (5) Returns an empty list for empty queries — no crash. The index uses `IndexFlatIP` (inner product on normalised vectors = cosine similarity). The index is built in `src/retrieval/index.py` → `build_index()`."

**Evidence:** `src/retrieval/retriever.py`, `src/retrieval/index.py`.

---

## Q15. What happens when the customer sends "still waiting"?

**Interviewer intent:** Can you trace a real edge case through the system?

**Strong answer:**
"'still waiting' gets cleaned to 'still waiting' (no mentions or URLs to remove). The classifier gets a 2-word input with no strong keyword signal — it will likely predict `other` or another intent with low confidence (~0.35-0.45). The retrieval will find some similar messages but with low similarity scores (~0.20) because 'still waiting' is too vague. The escalation layer will trigger on low confidence and escalate with the reason 'Low classifier confidence (0.38 < 0.60). Intent prediction is uncertain.' This is the correct behaviour — without context, we cannot safely auto-handle this message. It's documented as Failure Mode #1 in `reports/failure_analysis.md`."

**Evidence:** `src/decision/escalation.py`, `reports/failure_analysis.md` Failure #1.

---

## Q16. What would you improve with one more week?

**Interviewer intent:** Do you have genuine insight into your system's weaknesses?

**Strong answer:**
"Seven things: (1) Human labelling of 500+ examples to replace keyword auto-labels. (2) Conversation-aware classification — concatenate the previous message as context for short follow-ups. (3) Fine-tune the embedding model end-to-end with human-labelled data. (4) Human relevance ratings for retrieved examples to get a defensible Precision@K metric. (5) Probability calibration (Platt scaling) to make confidence scores more reliable. (6) Expand the always-escalate list based on which intents have the highest error rate. (7) A/B test template vs LLM reply to quantify the LLM's actual value-add."

**Evidence:** `reports/final_report.md` Section 12.
