# Final Submission Checklist

This checklist reflects the current repository state as of 2026-09-16.

Items marked as complete are verified in the codebase and data artifacts.
Items marked as pending require a human annotator and must not be fabricated.

## Required project items

- [x] Reproducible AppleSupport sample pipeline
  - Command: `python run_pipeline.py --brand AppleSupport --sample`
  - Output: `data/raw/sample_data.csv`
  - Status: script exists and sample generation is wired to AppleSupport.

- [x] AppleSupport sample dataset in the 500–1000 range
  - Current artifact: `data/raw/sample_data.csv`
  - Status: real sample rows are present; no fake data was inserted.

- [x] Golden-set template generation from real AppleSupport data
  - File: `data/golden/golden_200.json`
  - Status: human-label fields are intentionally left blank.

- [x] Golden-set codebook and sampling notes
  - Files: `data/golden/codebook.md`, `data/golden/intent_schema.json`
  - Status: present and aligned to AppleSupport.

- [x] Double-labelling template for 50 examples
  - File: `data/golden/double_label_50.csv`
  - Status: template exists; human labels and agreement values remain empty until annotated.

- [x] Majority + TF-IDF + embedding baseline comparison
  - Files: `reports/results/comparison.csv`, `reports/results/baselines_log.txt`
  - Status: comparison pipeline is present.

- [x] FAISS leakage audit logic
  - File: `scripts/build_retrieval.py`
  - Status: train/test/golden exclusion logic is implemented.

- [x] Evidence Gate: confidence + evidence + risk -> AUTO_HANDLE / ESCALATE
  - File: `src/decision/escalation.py`
  - Status: implemented.

- [x] Real open-source generation path with template fallback
  - Files: `src/generation/generator.py`, `src/generation/prompts.py`
  - Status: generation path is present; template is fallback only.

- [x] Evaluation harness for accuracy, macro F1, escalation metrics, and bootstrap CI
  - Files: `evaluate.py`, `src/evaluation/metrics.py`
  - Status: implemented.

- [x] LLM judge scaffold with temperature 0 and rubric dimensions
  - File: `src/evaluation/llm_judge.py`
  - Status: implemented.

- [x] Top-5 failure analysis template
  - File: `reports/failure_analysis.md`
  - Status: present.

- [x] README and report artifacts
  - Files: `README.md`, `reports/final_report.md`, `reports/decision_log.md`
  - Status: present and updated to AppleSupport.

- [ ] Human-labelled golden set with >=15 examples per intent including other
  - Status: pending human annotation; not fabricated.

- [ ] should_escalate on every row of the golden set
  - Status: pending human annotation; not fabricated.

- [ ] 50 double-labelled examples with Cohen's kappa
  - Status: pending human annotation; not fabricated.

- [ ] Final evaluation metrics with 95% bootstrap CI and real outputs
  - Status: pending actual human-labelled evaluation data.

- [ ] Human-vs-LLM agreement on 50 examples and kappa reporting
  - Status: pending actual human labels and judged examples.

- [ ] Final result numbers and real top-5 failure examples from evaluation runs
  - Status: pending actual labeled evaluation; no fake results inserted.

## Short submission note

The repository has been preserved and aligned to the AppleSupport specification without fabricating labels or metrics. Any remaining completion items require real human annotation and evaluation data, and should be filled in by a human before a final submission is claimed.
