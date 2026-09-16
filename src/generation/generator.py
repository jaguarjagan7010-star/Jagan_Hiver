"""
generator.py — LLM response generator with Ollama and HuggingFace backends.

Backend selection (set LLM_BACKEND in .env):
  "ollama"       — calls a local Ollama server (recommended, free, fast)
  "huggingface"  — downloads and runs a HF model locally (slower, larger)

Both backends return the same output schema:
{
    "reply":         str,    # the generated support reply
    "grounded":      bool,   # True if historical evidence was available
    "evidence_used": list    # list of historical examples that were provided
}
"""

import sys
import json
import re
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import LLM_BACKEND, OLLAMA_MODEL, OLLAMA_URL, HF_MODEL_NAME
from src.generation.prompts import build_generation_prompt


# ── Ollama backend ─────────────────────────────────────────────────────────────

def _call_ollama(prompt: str, model: str = OLLAMA_MODEL, url: str = OLLAMA_URL,
                 temperature: float = 0.1) -> str:
    """
    Call a local Ollama server.
    Pass temperature=0 for the LLM judge.
    """
    import requests

    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": 42,
        },
    }
    try:
        response = requests.post(
            f"{url}/api/generate",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama. Make sure Ollama is running:\n"
            "  1. Install from https://ollama.ai\n"
            "  2. Run: ollama serve\n"
            "  3. Run: ollama pull mistral"
        )


# ── HuggingFace backend ────────────────────────────────────────────────────────

_hf_pipeline = None   # cached to avoid reloading on every call

def _call_huggingface(prompt: str, model_name: str = HF_MODEL_NAME) -> str:
    """
    Run a local HuggingFace text-generation model.
    Default: TinyLlama-1.1B-Chat (~600 MB, fast on CPU).
    """
    global _hf_pipeline
    if _hf_pipeline is None:
        from transformers import pipeline
        print(f"Loading HuggingFace model: {model_name} (first call only)...")
        _hf_pipeline = pipeline(
            "text-generation",
            model=model_name,
            max_new_tokens=150,
            do_sample=False,
            temperature=1.0,   # required when do_sample=False
        )

    output = _hf_pipeline(prompt, max_new_tokens=150)[0]["generated_text"]
    if output.startswith(prompt):
        output = output[len(prompt):].strip()
    return output


# ── Fallback: template-based reply (no LLM needed) ────────────────────────────

def _template_reply(
    customer_message: str,
    intent: str,
    historical_examples: list[dict],
) -> str:
    """
    Template fallback when ALL LLM backends fail.
    Uses the most similar historical brand reply verbatim.
    """
    if historical_examples:
        best = historical_examples[0]
        return best["brand_text"]

    # Generic fallback by Apple-specific intent
    templates = {
        "account_access":   "We're sorry you're having trouble with your Apple ID. Please DM us and we'll help you regain access.",
        "billing_payment":  "We understand your concern about the charge. Please DM us your Apple ID and we'll review this for you.",
        "device_hardware":  "We're sorry to hear about your device issue. Please DM us your serial number and we'll look into repair options.",
        "software_update":  "We're sorry you're having trouble with the update. Please DM us your device model and iOS version.",
        "app_issue":        "We're sorry the app isn't working as expected. Please DM us with details and we'll assist you.",
        "icloud_sync":      "We're sorry you're having iCloud issues. Please DM us and we'll help you get synced up.",
        "connectivity":     "We're sorry you're having connectivity issues. Please DM us your device model and we'll troubleshoot together.",
        "warranty_repair":  "We can help with your warranty or repair question. Please DM us your serial number to get started.",
        "other":            "We're sorry to hear about your issue. Please DM us and we'll be happy to help.",
    }
    return templates.get(intent, "We're sorry to hear about your issue. Please DM us and we'll be happy to help.")


# ── Main generator ─────────────────────────────────────────────────────────────

def generate_reply(
    brand: str,
    customer_message: str,
    intent: str,
    confidence: float,
    historical_examples: list[dict],
    backend: str = LLM_BACKEND,
) -> dict:
    """
    Generate a support reply.

    Priority order:
      1. ollama  (real open-source LLM, requires `ollama serve`)
      2. huggingface (TinyLlama, downloads ~600 MB on first use)
      3. template (verbatim historical reply — fallback only)

    Returns:
        {
            "reply":         str,
            "grounded":      bool,
            "evidence_used": list[dict],
            "backend":       str,
        }
    """
    grounded = len(historical_examples) > 0

    if backend == "template":
        reply = _template_reply(customer_message, intent, historical_examples)
        return {"reply": reply, "grounded": grounded,
                "evidence_used": historical_examples, "backend": "template"}

    prompt = build_generation_prompt(
        brand=brand,
        customer_message=customer_message,
        intent=intent,
        confidence=confidence,
        historical_examples=historical_examples,
    )

    # Try primary backend
    try:
        if backend == "ollama":
            reply = _call_ollama(prompt)
        elif backend == "huggingface":
            reply = _call_huggingface(prompt)
        else:
            raise ValueError(f"Unknown backend: {backend}")
        return {"reply": reply, "grounded": grounded,
                "evidence_used": historical_examples, "backend": backend}
    except Exception as e:
        print(f"[generator] '{backend}' failed: {e}")

    # Fallback: try huggingface if ollama failed
    if backend == "ollama":
        try:
            print("[generator] Trying HuggingFace fallback...")
            reply = _call_huggingface(prompt)
            return {"reply": reply, "grounded": grounded,
                    "evidence_used": historical_examples, "backend": "huggingface_fallback"}
        except Exception as e2:
            print(f"[generator] HuggingFace fallback failed: {e2}")

    # Final fallback: template
    print("[generator] Using template fallback.")
    reply = _template_reply(customer_message, intent, historical_examples)
    return {"reply": reply, "grounded": grounded,
            "evidence_used": historical_examples, "backend": "template_fallback"}
