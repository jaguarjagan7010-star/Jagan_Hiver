# Data Leakage Audit

This document records all leakage checks performed and their outcomes.

---

## 1. Duplicate Tweet IDs in Raw Data

**Check:** `df["tweet_id"].duplicated().sum()`

**Where:** `scripts/explore_data.py` → `reports/results/dataset_summary.json`

**Status:** Checked. Any duplicates are dropped in `conversations.py` via
`df.drop_duplicates(subset="tweet_id", keep="first")`.

---

## 2. Duplicate Text Across Splits

**Check:** After splitting, verify no identical `clean_text` appears in both train and test.

**Where:** `scripts/train_baselines.py` → `load_and_prepare()`

**Mitigation:** Conversation-level split ensures all messages from the same conversation
go to the same split. Since near-duplicate messages typically come from the same conversation
(same customer, same issue), conversation-level splitting prevents most text-level near-duplicates.

**Residual risk:** Two different customers may send identical messages (e.g. "where is my order").
These are genuinely independent examples and are not leakage.

---

## 3. Same Conversation in Both Train and Test

**Check:** `assert len(train_convs & test_convs) == 0`

**Where:** `src/intents/labeling.py` → `split_conversations()` — assertion runs every time splits are created.

**Status:** ENFORCED. The assertion will raise an error if any conversation appears in both splits.

---

## 4. Retrieval Index Contains Test Data

**Check:** The FAISS index is built only from pairs whose `conversation_id` is in the train split.

**Where:** `scripts/build_retrieval.py` — explicitly filters:
```python
pairs_train = pairs[pairs["conv_str"].isin(train_conv_ids)]
```

**Status:** ENFORCED. Test conversations are never indexed.

**Consequence:** Some test queries will have low retrieval similarity because their exact
conversation is not in the index. This is correct and expected — it reflects real-world
performance where the system has not seen the exact query before.

---

## 5. Golden Set Contamination

**Check:** The golden set is sampled from the TEST split only.

**Where:** `src/evaluation/golden_set.py` → `build_golden_set()` takes `test_df` as input.

**Status:** ENFORCED. Golden set examples come from test data only, which was never used
for training or threshold tuning.

---

## 6. Test Data Used for Threshold Tuning

**Check:** Escalation thresholds are tuned using `escalation.tune_thresholds()` on the
VALIDATION set, not the test set.

**Where:** `src/decision/escalation.py` → `tune_thresholds()` takes validation data.

**Status:** ENFORCED. The test set is only touched once — during final evaluation.

---

## 7. Test Data Used for Model Selection (C Tuning)

**Check:** The regularisation parameter C for both TF-IDF+LR and Embedding+LR is selected
using validation macro F1, not test macro F1.

**Where:** `scripts/train_baselines.py` and `scripts/train_classifier.py` — both loop over
C values and evaluate on `X_val`, then train the final model and evaluate once on `X_test`.

**Status:** ENFORCED.

---

## 8. Near-Duplicate Detection

**Check:** We do not run explicit near-duplicate detection (e.g. MinHash LSH) because:
1. Conversation-level splitting handles the main source of near-duplicates.
2. Genuinely identical messages from different customers are independent examples.

**Residual risk:** Low. Documented here for transparency.

**Recommendation for one more week:** Run MinHash LSH on `clean_text` across splits
and remove any pairs with Jaccard similarity > 0.9.

---

## Summary

| Check | Method | Status |
|---|---|---|
| Duplicate tweet IDs | `drop_duplicates` in conversations.py | HANDLED |
| Same conversation in train/test | Assertion in labeling.py | ENFORCED |
| Retrieval index contains test data | Explicit filter in build_retrieval.py | ENFORCED |
| Golden set from test only | Input parameter in golden_set.py | ENFORCED |
| Threshold tuning on test | Validation set used in escalation.py | ENFORCED |
| C tuning on test | Validation set used in training scripts | ENFORCED |
| Near-duplicate text | Conversation-level split (partial mitigation) | PARTIAL |
