"""
prompts.py — Prompt templates for AppleSupport LLM response generator and judge.

Judge runs at temperature=0 for reproducibility.
"""

SYSTEM_PROMPT = """You are a customer support agent for {brand}.
Write a helpful, professional reply to the customer message below.

Rules:
1. Base your reply ONLY on the historical examples provided.
2. Never invent refund amounts, dates, policies, or promises not in the examples.
3. If examples lack enough information, say you will escalate to a specialist.
4. Keep the reply concise (2-4 sentences maximum).
5. Be empathetic and professional.
6. Do not mention that you are an AI or that you are using examples."""


def build_generation_prompt(
    brand: str,
    customer_message: str,
    intent: str,
    confidence: float,
    historical_examples: list[dict],
) -> str:
    system = SYSTEM_PROMPT.format(brand=brand)

    if historical_examples:
        examples_text = "\n\n".join([
            f"Example {i+1}:\n"
            f"  Customer: {ex['customer_text']}\n"
            f"  Support:  {ex['brand_text']}"
            for i, ex in enumerate(historical_examples[:5])
        ])
        evidence_section = f"Historical support examples:\n{examples_text}"
    else:
        evidence_section = "No historical examples available."

    return (
        f"{system}\n\n"
        f"Intent: {intent} (confidence: {confidence:.2f})\n\n"
        f"{evidence_section}\n\n"
        f'Customer message: "{customer_message}"\n\n'
        f"Support reply:"
    )


# ── LLM Judge (temperature=0 for reproducibility) ─────────────────────────────

JUDGE_CRITERIA = [
    "relevance",          # Does the reply address the customer's actual question?
    "grounding",          # Is the reply grounded in the historical examples?
    "helpfulness",        # Would this reply actually help the customer?
    "tone",               # Is the tone professional and empathetic?
    "no_hallucination",   # Does the reply avoid inventing facts? (5=no hallucination)
    "escalation_correct", # Is the escalation decision appropriate?
    "conciseness",        # Is the reply appropriately concise (2-4 sentences)?
]


def build_judge_prompt(
    customer_message: str,
    generated_reply: str,
    historical_examples: list[dict],
    intent: str,
    decision: str = "",
) -> str:
    """
    Build the LLM-as-a-judge prompt.
    Judge must run at temperature=0 for reproducibility.
    Scores each of 7 criteria 1-5.
    """
    examples_text = "\n".join([
        f"  [{i+1}] Customer: {ex['customer_text'][:100]} | Support: {ex['brand_text'][:100]}"
        for i, ex in enumerate(historical_examples[:3])
    ]) if historical_examples else "  None available."

    criteria_list = "\n".join([
        f'  "{c}": {{"score": <1-5>, "reason": "<one sentence>"}}'
        for c in JUDGE_CRITERIA
    ])

    return (
        f'You are evaluating an AppleSupport reply. Score each criterion 1 (poor) to 5 (excellent).\n\n'
        f'Customer message: "{customer_message}"\n'
        f'Detected intent: {intent}\n'
        f'Decision: {decision}\n'
        f'Historical examples used:\n{examples_text}\n\n'
        f'Generated reply: "{generated_reply}"\n\n'
        f'Respond ONLY with valid JSON in this exact format:\n'
        f'{{\n{criteria_list}\n}}'
    )
