"""
Utility functions for APTBench Agentic Math tasks.

These tasks evaluate mathematical atomic capabilities including:
- Symbol Recognition, Concept Understanding, Calculation Execution
- Spatial Perception, Formal Math Language, Multi-Step Reasoning
- Proof Construction, Theorem Application, Modeling Transformation
- New Knowledge Learning
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


# ============================================================================
# Prompt Building Functions (doc_to_text)
# ============================================================================

def doc_to_text_planning_single(doc: Dict[str, Any]) -> str:
    """
    Build prompt for planning_single task.

    This task asks which mathematical atomic capability is most critical
    for solving a given math question. Uses 3-shot learning.
    """
    template = _load_prompt_template("math_planning_single_3shot.txt")
    prompt = template.replace("$REPO_INFO$", doc["repo_info"])
    prompt = prompt.replace("$CHOICES$", doc["choices"])
    return prompt


def doc_to_text_feedback_tf(doc: Dict[str, Any]) -> str:
    """
    Build prompt for feedback_tf task.

    This task judges whether a proposed solution to a math problem
    is correct or wrong.
    """
    template = _load_prompt_template("math_feedback_tf.txt")
    prompt = template.replace("$REPO_INFO$", doc["repo_info"])
    return prompt


def doc_to_text_action_cal(doc: Dict[str, Any]) -> str:
    """
    Build prompt for action_cal task.

    This task asks to solve a math calculation problem and choose
    the correct answer from multiple choices.
    """
    template = _load_prompt_template("math_action_calculation.txt")
    prompt = template.replace("$REPO_INFO$", doc["repo_info"])
    prompt = prompt.replace("$CHOICES$", doc["choices"])
    return prompt


# ============================================================================
# Answer Extraction Functions
# ============================================================================

def extract_answer_choice_ae(response: str) -> str:
    """
    Extract answer choice from A-E for planning_single task.

    Looks for patterns like:
    - "A)", "B ", "C:", etc.
    - Letters at end of response

    Returns uppercase letter or empty string if not found.
    """
    if not response:
        return ""

    # Remove asterisks that might be in bold formatting
    response = response.replace('*', '')

    # Look for letter followed by ), newline, space, or colon
    match = re.search(r'\b([A-E])[\)\n\s:]', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Try letter at end of response
    match = re.search(r'\b([A-E])\s*$', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return ""


def extract_answer_choice_ad(response: str) -> str:
    """
    Extract answer choice from A-D for action_cal task.

    Similar to extract_answer_choice_ae but for A-D range.
    """
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


def extract_answer_true_false(response: str) -> str:
    """
    Extract correct/wrong answer for feedback_tf task.

    Looks for "correct" or "wrong" in the response.
    Returns lowercase "correct" or "wrong", or empty string if not found.
    """
    if not response:
        return ""

    response = response.lower().replace('*', '')

    match = re.search(r'\b(correct|wrong)\b', response)
    if match:
        return match.group(1)

    return ""


# ============================================================================
# Target Extraction Functions (doc_to_target)
# ============================================================================

def doc_to_target_planning_single(doc: Dict[str, Any]) -> str:
    """Get ground truth answer for planning_single task."""
    return doc["answer"]


def doc_to_target_feedback_tf(doc: Dict[str, Any]) -> str:
    """Get ground truth answer for feedback_tf task."""
    return doc["answer"]


def doc_to_target_action_cal(doc: Dict[str, Any]) -> str:
    """Get ground truth answer for action_cal task."""
    return doc["answer"]


# ============================================================================
# Result Processing Functions
# ============================================================================

def process_results_planning_single(doc: Dict[str, Any], results: list) -> Dict[str, Any]:
    """
    Process results for planning_single task.

    Compares extracted answer (A-E) with ground truth.
    """
    # results is a list of model outputs
    response = results[0] if results else ""

    pred = extract_answer_choice_ae(response)
    gold = doc["answer"].upper()

    return {
        "acc": float(pred == gold),
        "exact_match": float(pred == gold),
    }


def process_results_feedback_tf(doc: Dict[str, Any], results: list) -> Dict[str, Any]:
    """
    Process results for feedback_tf task.

    Compares extracted answer (correct/wrong) with ground truth.
    """
    response = results[0] if results else ""

    pred = extract_answer_true_false(response)
    gold = doc["answer"].lower()

    return {
        "acc": float(pred == gold),
        "exact_match": float(pred == gold),
    }


def process_results_action_cal(doc: Dict[str, Any], results: list) -> Dict[str, Any]:
    """
    Process results for action_cal task.

    Compares extracted answer (A-D) with ground truth.
    """
    response = results[0] if results else ""

    pred = extract_answer_choice_ad(response)
    gold = doc["answer"].upper()

    return {
        "acc": float(pred == gold),
        "exact_match": float(pred == gold),
    }
