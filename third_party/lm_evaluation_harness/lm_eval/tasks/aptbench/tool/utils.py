"""
Utility functions for APTBench Tool Use tasks.

These tasks evaluate function calling and API usage capabilities:
- Selecting the correct API/function from a list
- Generating appropriate parameters for function calls
"""

import re
import json
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

def doc_to_text_acebench_api_select(doc: Dict[str, Any]) -> str:
    """Select the correct tool function name from the given tool function list."""
    return _build_prompt("tool_acebench_api_select_3shot.txt", {
        "$USER_PROMPT$": doc["user_prompt"],
        "$FUNCS$": doc["function"]
    })


def doc_to_text_acebench_api_param(doc: Dict[str, Any]) -> str:
    """Write the correct parameters to pass according to user query and selected function."""
    return _build_prompt("tool_acebench_api_param_3shot.txt", {
        "$USER_PROMPT$": doc["user_prompt"],
        "$FUNCS$": doc["function"],
        "$ANSWER_SELECT$": doc["answer4select"],
        "$PARA_NAME$": doc.get("param_name", "")
    })


def doc_to_text_bfcl_v4_api_select(doc: Dict[str, Any]) -> str:
    """Select the correct tool function name from the given tool function list (BFCL v4)."""
    return _build_prompt("tool_bfcl_v4_api_select_3shot.txt", {
        "$USER_PROMPT$": doc["user_prompt"],
        "$FUNCS$": doc["function"]
    })


def doc_to_text_bfcl_v4_api_param(doc: Dict[str, Any]) -> str:
    """Write the correct parameters to pass according to user query and selected function (BFCL v4)."""
    return _build_prompt("tool_bfcl_v4_api_param_3shot.txt", {
        "$USER_PROMPT$": doc["user_prompt"],
        "$FUNCS$": doc["function"],
        "$ANSWER_SELECT$": doc["answer4select"],
        "$PARA_NAME$": doc.get("param_name", "")
    })


# ============================================================================
# Answer Extraction Functions
# ============================================================================

def extract_from_code_block(response: str) -> str:
    """
    Extract content from code block (```...```).
    Returns the content inside the code block, or the trimmed response if no block found.
    """
    if not response:
        return ""

    response = response.replace('*', '')

    # Try to extract from code block
    code_block_match = re.search(r'```(?:\w+)?\s*\n?(.+?)```', response, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

    return response.strip()


# ============================================================================
# Target Extraction (all tasks use answer4select or answer4param)
# ============================================================================

def doc_to_target_api_select(doc: Dict[str, Any]) -> str:
    """Get ground truth function name for api_select tasks."""
    return doc["answer4select"]


def doc_to_target_api_param(doc: Dict[str, Any]) -> str:
    """Get ground truth parameters for api_param tasks."""
    return doc["answer4param"]


# Aliases for consistency with YAML !function references
doc_to_target_acebench_api_select = doc_to_target_api_select
doc_to_target_acebench_api_param = doc_to_target_api_param
doc_to_target_bfcl_v4_api_select = doc_to_target_api_select
doc_to_target_bfcl_v4_api_param = doc_to_target_api_param


# ============================================================================
# Result Processing Functions
# ============================================================================

def _process_api_select(doc: Dict[str, Any], results: list) -> Dict[str, Any]:
    """Process results for API selection tasks (function name matching)."""
    response = results[0] if results else ""
    pred = extract_from_code_block(response)
    gold = doc["answer4select"]

    # Normalize whitespace
    pred_normalized = " ".join(pred.split())
    gold_normalized = " ".join(gold.split())

    return {
        "acc": float(pred_normalized == gold_normalized),
        "exact_match": float(pred_normalized == gold_normalized),
    }


def _process_api_param(doc: Dict[str, Any], results: list) -> Dict[str, Any]:
    """Process results for API parameter generation tasks (JSON matching)."""
    response = results[0] if results else ""
    pred_str = extract_from_code_block(response)
    gold_str = doc["answer4param"]

    # Try to parse both as JSON and compare
    # If parsing fails, fall back to string comparison
    try:
        pred_json = json.loads(pred_str)
        gold_json = json.loads(gold_str)
        match = pred_json == gold_json
    except (json.JSONDecodeError, TypeError):
        # Fall back to normalized string comparison
        pred_normalized = " ".join(pred_str.split())
        gold_normalized = " ".join(gold_str.split())
        match = pred_normalized == gold_normalized

    return {
        "acc": float(match),
        "exact_match": float(match),
    }


# Aliases for specific tasks
process_results_acebench_api_select = _process_api_select
process_results_acebench_api_param = _process_api_param
process_results_bfcl_v4_api_select = _process_api_select
process_results_bfcl_v4_api_param = _process_api_param
