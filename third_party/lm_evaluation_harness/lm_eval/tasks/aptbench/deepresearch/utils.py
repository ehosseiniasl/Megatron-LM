"""
Utility functions for APTBench Deep Research tasks.

These tasks evaluate research capabilities including:
- Planning research steps
- Summarizing answers from search trajectories
- Choosing report structures
- Generating citations
- Evaluating response quality

Tasks are bilingual (English and Chinese).
"""

import re
import os
from typing import Dict, Any


# Cache for loaded prompt templates
_PROMPT_CACHE = {}


def _load_prompt_template(filename: str) -> str:
    """Load a prompt template file from the same directory as this module."""
    if filename not in _PROMPT_CACHE:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(current_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            _PROMPT_CACHE[filename] = f.read()
    return _PROMPT_CACHE[filename]


def _build_prompt(template_file: str, replacements: Dict[str, str]) -> str:
    """Generic prompt builder that loads template and applies replacements."""
    template = _load_prompt_template(template_file)
    prompt = template
    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, value)
    return prompt


# ============================================================================
# Prompt Building Functions (doc_to_text)
# ============================================================================

def doc_to_text_plan_en(doc: Dict[str, Any]) -> str:
    """Determine which research step is correct (English)."""
    return _build_prompt("deepresearch_plan_en_3shot.txt", {
        "$QUERY$": doc["query"],
        "$TRAJ$": doc["trajectory"],
        "$CHOICES$": doc["choices"]
    })


def doc_to_text_plan_zh(doc: Dict[str, Any]) -> str:
    """Determine which research step is correct (Chinese)."""
    return _build_prompt("deepresearch_plan_zh_3shot.txt", {
        "$QUERY$": doc["query"],
        "$TRAJ$": doc["trajectory"],
        "$CHOICES$": doc["choices"]
    })


def doc_to_text_summ_ans_en(doc: Dict[str, Any]) -> str:
    """Summarize answer from research trajectory (English)."""
    return _build_prompt("deepresearch_summ_ans_en_3shot.txt", {
        "$QUERY$": doc["query"],
        "$TRAJ$": doc["trajectory"]
    })


def doc_to_text_summ_ans_zh(doc: Dict[str, Any]) -> str:
    """Summarize answer from research trajectory (Chinese)."""
    return _build_prompt("deepresearch_summ_ans_zh_3shot.txt", {
        "$QUERY$": doc["query"],
        "$TRAJ$": doc["trajectory"]
    })


def doc_to_text_openend_plan_en(doc: Dict[str, Any]) -> str:
    """Choose best report structure for a research question."""
    return _build_prompt("deepresearch_openend_plan_en_3shot.txt", {
        "$QUERY$": doc["question"],
        "$CHOICES$": doc["choices"]
    })


def doc_to_text_openend_citation_en(doc: Dict[str, Any]) -> str:
    """Choose correct citation for an article (English)."""
    return _build_prompt("deepresearch_openend_citation_en_3shot.txt", {
        "$QUERY$": doc["query"],
        "$ARTICLE$": doc["article"],
        "$URL$": doc["url"],
        "$URL_CONTENT$": doc["url_content"],
        "$CHOICES$": doc["choices"]
    })


def doc_to_text_openend_citation_zh(doc: Dict[str, Any]) -> str:
    """Choose correct citation for an article (Chinese)."""
    return _build_prompt("deepresearch_openend_citation_zh_3shot.txt", {
        "$QUERY$": doc["query"],
        "$ARTICLE$": doc["article"],
        "$URL$": doc["url"],
        "$URL_CONTENT$": doc["url_content"],
        "$CHOICES$": doc["choices"]
    })


def doc_to_text_openend_quality_en(doc: Dict[str, Any]) -> str:
    """Evaluate quality of research response (English)."""
    return _build_prompt("deepresearch_openend_quality_en_2shot.txt", {
        "$QUERY$": doc["query"],
        "$CHOICES$": doc["choices"]
    })


def doc_to_text_openend_quality_zh(doc: Dict[str, Any]) -> str:
    """Evaluate quality of research response (Chinese)."""
    return _build_prompt("deepresearch_openend_quality_zh_2shot.txt", {
        "$QUERY$": doc["query"],
        "$CHOICES$": doc["choices"]
    })


# ============================================================================
# Answer Extraction Functions
# ============================================================================

def extract_answer_choice_ag(response: str) -> str:
    """Extract answer choice from A-G for plan tasks."""
    if not response:
        return ""

    response = response.replace('*', '')

    # Look for letter followed by ), newline, space, or colon
    match = re.search(r'\b([A-G])[\)\n\s:]', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Try letter at end of response
    match = re.search(r'\b([A-G])\s*$', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return ""


def extract_answer_choice_ad(response: str) -> str:
    """Extract answer choice from A-D for citation and quality tasks."""
    if not response:
        return ""

    response = response.replace('*', '')

    match = re.search(r'\b([A-D])[\)\n\s:]', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r'\b([A-D])\s*$', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return ""


def extract_answer_from_brackets(response: str) -> str:
    """Extract answer from square brackets [answer]."""
    if not response:
        return ""

    response = response.replace('*', '')

    # Look for content in square brackets
    match = re.search(r'\[([^\]]+)\]', response)
    if match:
        return match.group(1).strip()

    return response.strip()


# ============================================================================
# Target Extraction
# ============================================================================

def doc_to_target(doc: Dict[str, Any]) -> str:
    """Get ground truth answer (same for all deepresearch tasks)."""
    return doc["answer"]


# Aliases for consistency
doc_to_target_plan_en = doc_to_target
doc_to_target_plan_zh = doc_to_target
doc_to_target_summ_ans_en = doc_to_target
doc_to_target_summ_ans_zh = doc_to_target
doc_to_target_openend_plan_en = doc_to_target
doc_to_target_openend_citation_en = doc_to_target
doc_to_target_openend_citation_zh = doc_to_target
doc_to_target_openend_quality_en = doc_to_target
doc_to_target_openend_quality_zh = doc_to_target


# ============================================================================
# Result Processing Functions
# ============================================================================

def _process_multiple_choice_ag(doc: Dict[str, Any], results: list) -> Dict[str, Any]:
    """Process multiple choice tasks (A-G)."""
    response = results[0] if results else ""
    pred = extract_answer_choice_ag(response)
    gold = doc["answer"].upper()

    return {
        "acc": float(pred == gold),
        "exact_match": float(pred == gold),
    }


def _process_multiple_choice_ad(doc: Dict[str, Any], results: list) -> Dict[str, Any]:
    """Process multiple choice tasks (A-D)."""
    response = results[0] if results else ""
    pred = extract_answer_choice_ad(response)
    gold = doc["answer"].upper()

    return {
        "acc": float(pred == gold),
        "exact_match": float(pred == gold),
    }


def _process_text_answer(doc: Dict[str, Any], results: list) -> Dict[str, Any]:
    """Process free text answer extraction tasks (summ_ans)."""
    response = results[0] if results else ""
    pred = extract_answer_from_brackets(response)
    gold = doc["answer"]

    # Normalize whitespace for comparison
    pred_normalized = " ".join(pred.split()).lower()
    gold_normalized = " ".join(gold.split()).lower()

    return {
        "acc": float(pred_normalized == gold_normalized),
        "exact_match": float(pred_normalized == gold_normalized),
    }


# Aliases for specific tasks
process_results_plan_en = _process_multiple_choice_ag
process_results_plan_zh = _process_multiple_choice_ag
process_results_summ_ans_en = _process_text_answer
process_results_summ_ans_zh = _process_text_answer
process_results_openend_plan_en = _process_multiple_choice_ag
process_results_openend_citation_en = _process_multiple_choice_ad
process_results_openend_citation_zh = _process_multiple_choice_ad
process_results_openend_quality_en = _process_multiple_choice_ad
process_results_openend_quality_zh = _process_multiple_choice_ad
