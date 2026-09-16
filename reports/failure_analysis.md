# Failure Analysis

Top 5 failure modes identified for review from the existing evaluation workflow.
The examples below are illustrative templates until human-labelled evaluation is complete;
they are not presented as new measured results.

> **Note:** Run `python scripts/evaluate.py` first to populate real examples.
> The structure below is the template; examples will be filled from actual errors.

---

## Failure Mode 1: Short Follow-Up Messages

**Example customer message:** `"still waiting"`

**Expected intent:** `other` or the intent established by the preceding conversation, which is unavailable here

**Predicted intent:** Low-confidence prediction among the canonical intents

**Classifier confidence:** ~0.35 (low)

**Retrieved evidence:** Low similarity scores (~0.20) — no historical example matches "still waiting" well.

**Generated response:** Generic fallback reply.

**Why it failed:** The message has no content beyond two words. Without conversation context (the previous message that established the order issue), the classifier has no signal. TF-IDF finds no matching keywords. The embedding model cannot infer intent from two words alone.

**Hypothesis:** Short follow-up messages are inherently ambiguous without conversation context. The system processes each message independently and has no memory of prior turns.

**Possible improvement:** Include the previous message in the conversation as additional context when classifying. Build a conversation-aware classifier that concatenates the last N turns.

---

## Failure Mode 2: Overlapping Billing and Warranty/Repair Signals

**Example customer message:** `"My damaged iPhone needs repair, and I want to know whether I can get a refund"`

**Expected intent:** `billing_payment` or `warranty_repair`, depending on the customer's primary request

**Predicted intent:** One of the two overlapping canonical intents

**Classifier confidence:** ~0.52 (borderline)

**Retrieved evidence:** Mixed — some billing examples and some warranty/repair examples.

**Generated response:** Focuses on refund process, ignores return instructions.

**Why it failed:** The message contains signals for both a payment issue and a repair question. A single-label classifier must choose one primary intent, and the auto-labeler may not capture that ambiguity.

**Hypothesis:** Overlapping resolution paths create systematic ambiguity when a customer combines financial and physical-device concerns.

**Possible improvement:** Preserve the `other` fallback and flag ambiguous multi-intent messages for escalation or human review. Do not treat auto-labels as human ground truth.

---

## Failure Mode 3: Brand-Specific Jargon

**Example customer message:** `"My iPhone is not connecting to my Wi-Fi"`

**Expected intent:** `connectivity`

**Predicted intent:** `connectivity` or another low-confidence canonical intent

**Classifier confidence:** ~0.48

**Retrieved evidence:** Moderate similarity (~0.35).

**Generated response:** Asks for product details rather than troubleshooting steps.

**Why it failed:** Device-specific wording can be sparse or expressed indirectly. The keyword rules and TF-IDF model may not see enough examples of a particular Apple device or connection failure.

**Hypothesis:** Rare device names and indirect connectivity wording remain difficult for lexical models.

**Possible improvement:** Add more human-labelled AppleSupport examples for rare device and connectivity wording in a future iteration.

---

## Failure Mode 4: Hallucinated Policy in LLM Response

**Example customer message:** `"How long does a refund take?"`

**Expected intent:** `billing_payment` (→ ESCALATE by policy)

**Predicted intent:** `billing_payment` ✓

**Decision:** ESCALATE ✓ (correct)

**Generated response (template fallback):** Retrieved a historical reply that mentions "3-5 business days" — but this may be outdated or incorrect for the customer's specific payment method.

**Why it failed:** The template fallback returns the most similar historical reply verbatim. Historical replies may contain specific timeframes that are no longer accurate or that vary by payment method.

**Hypothesis:** Historical brand replies are not always accurate ground truth. They reflect what a support agent said at a specific time, which may be outdated.

**Possible improvement:** Flag responses that contain specific numbers (days, amounts) as requiring human review. Add a post-processing step that detects and redacts potentially outdated specifics.

---

## Failure Mode 5: Sarcasm and Indirect Complaints

**Example customer message:** `"Great, another update that failed. Really loving this iPhone right now."`

**Expected intent:** `software_update` or `other`, depending on whether the update failure is explicit

**Predicted intent:** `other` or another low-confidence canonical intent

**Classifier confidence:** ~0.55

**Retrieved evidence:** Low similarity — sarcastic phrasing doesn't match direct complaint examples.

**Generated response:** Generic "thank you for your feedback" response — completely wrong.

**Why it failed:** The message uses sarcasm, which can obscure the actual software-update complaint and make it resemble a general complaint.

**Hypothesis:** Neither TF-IDF nor sentence embeddings reliably handle sarcasm or indirect complaints.

**Possible improvement:** Add a sarcasm/negation detection pre-processing step. Train on more examples of indirect complaints. Use a sentiment classifier as an additional feature.

---

## Summary Table

| # | Failure Mode | Frequency | Severity | Fix Complexity |
|---|---|---|---|---|
| 1 | Short follow-ups without context | High | Medium | Medium |
| 2 | Overlapping intents (return/refund) | Medium | Medium | Low |
| 3 | Brand-specific jargon not in rules | Medium | Low | Low |
| 4 | Outdated specifics in historical replies | Low | High | Medium |
| 5 | Sarcasm / indirect complaints | Low | Medium | High |
