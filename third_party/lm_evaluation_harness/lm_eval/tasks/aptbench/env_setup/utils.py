"""
Utility functions for APTBench Environment Setup tasks.

These tasks evaluate software engineering capabilities for environment configuration:
- Planning execution steps for repository setup
- Diagnosing and fixing setup errors
- Predicting next execution commands
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

def doc_to_text_plan(doc: Dict[str, Any]) -> str:
    """
    Build prompt for env_setup plan task.

    Choose the correct execution plan (A-G) to setup the environment
    based on repository file analysis.
    """
    template = _load_prompt_template("env_setup_plan_3shot.txt")
    prompt = template.replace("$REPO_INFO$", doc["repo_info"])
    prompt = prompt.replace("$CHOICES$", doc["choices"])
    return prompt


def doc_to_text_error(doc: Dict[str, Any]) -> str:
    """
    Build prompt for env_setup error task.

    Choose the correct solution (A-G) to fix a setup error given
    the setup instructions, issue title, and issue body.
    """
    template = _load_prompt_template("env_setup_error_3shot.txt")
    prompt = template.replace("$SETUP$", doc["setup_instruct"])
    prompt = prompt.replace("$ISSUE_TITLE$", doc["issue_title"])
    prompt = prompt.replace("$ISSUE_BODY$", doc["issue_body"])
    prompt = prompt.replace("$CHOICES$", doc["choices"])
    return prompt


def doc_to_text_action(doc: Dict[str, Any]) -> str:
    """
    Build prompt for env_setup action task.

    Predict the next execution command given an execution plan
    and previously executed commands.
    """
    template = _load_prompt_template("env_setup_action_3shot.txt")
    prompt = template.replace("$EXE_PLAN$", doc["execution_plan"])
    prompt = prompt.replace("$EXE_CMDS$", doc["executed_cmds"])
    return prompt


# ============================================================================
# Answer Extraction Functions
# ============================================================================

def extract_answer_choice(response: str) -> str:
    """
    Extract answer choice from A-G for plan and error tasks.

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
    match = re.search(r'\b([A-G])[\)\n\s:]', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Try letter at end of response
    match = re.search(r'\b([A-G])\s*$', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return ""


def extract_answer_action(response: str) -> str:
    """
    Extract command from action task response.

    The response should contain a bash command, potentially wrapped
    in ```bash``` code blocks. Extract the actual command.
    """
    if not response:
        return ""

    # Remove asterisks
    response = response.replace('*', '')

    # Try to extract from code block
    code_block_match = re.search(r'```(?:bash)?\s*\n?(.+?)```', response, re.DOTALL)
    if code_block_match:
        cmd = code_block_match.group(1).strip()
        return cmd

    # If no code block, return the trimmed response
    return response.strip()


# ============================================================================
# Target Extraction Functions (doc_to_target)
# ============================================================================

def doc_to_target_plan(doc: Dict[str, Any]) -> str:
    """Get ground truth answer for plan task."""
    return doc["answer"]


def doc_to_target_error(doc: Dict[str, Any]) -> str:
    """Get ground truth answer for error task."""
    return doc["answer"]


def doc_to_target_action(doc: Dict[str, Any]) -> str:
    """Get ground truth answer for action task."""
    # Construct the full command from prefix + target + postfix
    prefix = doc.get("cmd_prefix", "").strip()
    target = doc["target_cmd"].strip()
    postfix = doc.get("cmd_postfix", "").strip()

    # Build the command
    if prefix and postfix:
        full_cmd = f"{prefix} {postfix}"
    elif prefix:
        full_cmd = prefix
    elif postfix:
        full_cmd = postfix
    else:
        full_cmd = target

    return full_cmd


# ============================================================================
# Result Processing Functions
# ============================================================================

def process_results_plan(doc: Dict[str, Any], results: list) -> Dict[str, Any]:
    """
    Process results for plan task.

    Compares extracted answer (A-G) with ground truth.
    """
    response = results[0] if results else ""

    pred = extract_answer_choice(response)
    gold = doc["answer"].upper()

    return {
        "acc": float(pred == gold),
        "exact_match": float(pred == gold),
    }


def process_results_error(doc: Dict[str, Any], results: list) -> Dict[str, Any]:
    """
    Process results for error task.

    Compares extracted answer (A-G) with ground truth.
    """
    response = results[0] if results else ""

    pred = extract_answer_choice(response)
    gold = doc["answer"].upper()

    return {
        "acc": float(pred == gold),
        "exact_match": float(pred == gold),
    }


def process_results_action(doc: Dict[str, Any], results: list) -> Dict[str, Any]:
    """
    Process results for action task.

    Compares extracted command with ground truth command.
    Uses exact string match after normalization.
    """
    response = results[0] if results else ""

    pred = extract_answer_action(response)
    gold = doc_to_target_action(doc)

    # Normalize whitespace for comparison
    pred_normalized = " ".join(pred.split())
    gold_normalized = " ".join(gold.split())

    return {
        "acc": float(pred_normalized == gold_normalized),
        "exact_match": float(pred_normalized == gold_normalized),
    }
