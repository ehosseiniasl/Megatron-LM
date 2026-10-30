"""
Utility functions for APTBench Issue Fix tasks.

These tasks evaluate software engineering capabilities for bug fixing:
- Locating bugs in code
- Developing patches to fix issues
- Planning fix solutions
- Executing fixes
- Validating patches with tests
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

def doc_to_text_locate(doc: Dict[str, Any]) -> str:
    """Select the code snippet generating the bug."""
    return _build_prompt("issue_fix_locate_3shot.txt", {
        "$ISSUE$": doc["issue_statement"],
        "$CHOICES$": doc["choices"]
    })


def doc_to_text_fix_patch(doc: Dict[str, Any]) -> str:
    """Choose the correct patch to fix the issue."""
    return _build_prompt("issue_fix_fix_patch_3shot.txt", {
        "$ISSUE$": doc["issue_statement"],
        "$CHOICES$": doc["choices"]
    })


def doc_to_text_plan(doc: Dict[str, Any]) -> str:
    """Choose the correct plan to solve the issue."""
    return _build_prompt("issue_fix_plan_3shot.txt", {
        "$TRAJ$": doc["trajs"],
        "$CHOICES$": doc["choices"]
    })


def doc_to_text_action(doc: Dict[str, Any]) -> str:
    """Write the next execution command from the tool set."""
    return _build_prompt("issue_fix_action_3shot.txt", {
        "$TRAJ$": doc["trajs"]
    })


def doc_to_text_test_patch(doc: Dict[str, Any]) -> str:
    """Choose the correct test to validate the patch."""
    return _build_prompt("issue_fix_test_patch_3shot.txt", {
        "$INSTRUCT$": doc["question_instruct"],
        "$PROBLEM$": doc["problem_statement"],
        "$CHOICES$": doc["choices"]
    })


# ============================================================================
# Answer Extraction Functions
# ============================================================================

def extract_answer_choice_ad(response: str) -> str:
    """Extract answer choice from A-D for multiple choice tasks."""
    if not response:
        return ""

    response = response.replace('*', '')

    # Look for letter followed by ), newline, space, or colon
    match = re.search(r'\b([A-D])[\)\n\s:]', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Try letter at end of response
    match = re.search(r'\b([A-D])\s*$', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return ""


def extract_answer_action(response: str) -> str:
    """Extract command from action task response."""
    if not response:
        return ""

    response = response.replace('*', '')

    # Try to extract from code block
    code_block_match = re.search(r'```(?:bash)?\s*\n?(.+?)```', response, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

    return response.strip()


# ============================================================================
# Target Extraction (all tasks just return doc["answer"])
# ============================================================================

def doc_to_target(doc: Dict[str, Any]) -> str:
    """Get ground truth answer (same for all issue_fix tasks)."""
    return doc["answer"]


# Aliases for consistency with YAML !function references
doc_to_target_locate = doc_to_target
doc_to_target_fix_patch = doc_to_target
doc_to_target_plan = doc_to_target
doc_to_target_action = doc_to_target
doc_to_target_test_patch = doc_to_target


# ============================================================================
# Result Processing Functions
# ============================================================================

def _process_multiple_choice(doc: Dict[str, Any], results: list) -> Dict[str, Any]:
    """Generic result processor for multiple choice tasks (A-D)."""
    response = results[0] if results else ""
    pred = extract_answer_choice_ad(response)
    gold = doc["answer"].upper()

    return {
        "acc": float(pred == gold),
        "exact_match": float(pred == gold),
    }


def process_results_action(doc: Dict[str, Any], results: list) -> Dict[str, Any]:
    """Process results for action task (command extraction)."""
    response = results[0] if results else ""
    pred = extract_answer_action(response)
    gold = doc["answer"]

    # Normalize whitespace for comparison
    pred_normalized = " ".join(pred.split())
    gold_normalized = " ".join(gold.split())

    return {
        "acc": float(pred_normalized == gold_normalized),
        "exact_match": float(pred_normalized == gold_normalized),
    }


# Aliases for multiple choice tasks (all use the same logic)
process_results_locate = _process_multiple_choice
process_results_fix_patch = _process_multiple_choice
process_results_plan = _process_multiple_choice
process_results_test_patch = _process_multiple_choice
