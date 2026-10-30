from lm_eval.filters.sanitize import sanitize
from lm_eval.metrics.code_eval import check_correctness
from lm_eval.metrics.code_eval import evaluate_candidates


def build_text_complete(doc: dict):
    """Build the prompt for the Complete split.
    
    In BigCodeBench Complete, the model receives the function signature
    and docstring, and must generate the complete implementation.
    """
    return doc["complete_prompt"]


def build_text_instruct(doc: dict):
    """Build the prompt for the Instruct split.
    
    In BigCodeBench Instruct, the model receives a natural language description
    and must generate the complete self-contained solution.
    
    This mimics the Skills implementation by extracting the instruction prefix
    and combining it with the code prompt.
    """
    # Extract the instruction part before the delimiter
    instruct_prompt = doc["instruct_prompt"]
    delimiter = "You should write self-contained code starting with:"
    
    # Find where the delimiter starts
    delimiter_idx = instruct_prompt.find(delimiter)
    if delimiter_idx == -1:
        # If delimiter not found, use full instruct_prompt
        return instruct_prompt
    
    # Extract prefix (instruction) and the code prompt part
    prefix = instruct_prompt[:delimiter_idx].strip()
    
    # Get the code prompt and wrap in markdown if not already wrapped
    code_prompt = doc.get("code_prompt", "")
    if code_prompt and "```" not in code_prompt:
        code_prompt = f"```python\n{code_prompt}\n```"
    
    # Reconstruct the prompt similar to Skills
    return f"{prefix}\n\n{delimiter}\n{code_prompt}"


def build_references(doc: dict):
    """Build test program from test cases.
    
    BigCodeBench provides test cases that can be directly executed
    to verify the correctness of generated code.
    """
    return doc["test"]


def process_resp(repeat_idx: int, resp: str, doc: dict):
    """Process a single response by sanitizing and executing tests.
    
    Args:
        repeat_idx: Index of the repeat (for pass@k estimation)
        resp: Generated code response from the model
        doc: Document containing the problem and tests
        
    Returns:
        Dictionary with execution results including pass/fail status
    """
    # For Complete split, combine the prompt with the generated completion
    candidate_code = sanitize(
        doc["complete_prompt"] + resp, 
        entrypoint=doc["entry_point"]
    )
    
    # Build the complete test program
    test_program = candidate_code + "\n" + build_references(doc)
    
    # Execute the test program
    result = check_correctness(
        test_program, 
        timeout=5,  # BigCodeBench may need more time than HumanEval
        task_id=doc["task_id"], 
        completion_id=repeat_idx
    )
    return result


def process_resp_instruct(repeat_idx: int, resp: str, doc: dict):
    """Process a single response for instruct mode by sanitizing and executing tests.
    
    In instruct mode, the model generates the complete solution from a natural
    language description, so we don't prepend any prompt - the response should
    be the full code.
    
    Args:
        repeat_idx: Index of the repeat (for pass@k estimation)
        resp: Generated code response from the model (full solution)
        doc: Document containing the problem and tests
        
    Returns:
        Dictionary with execution results including pass/fail status
    """
    # For Instruct mode, the response should be the complete solution
    candidate_code = sanitize(
        resp,  # No prompt prepended - model generates full solution
        entrypoint=doc["entry_point"]
    )
    
    # Build the complete test program
    test_program = candidate_code + "\n" + build_references(doc)
    
    # Execute the test program
    result = check_correctness(
        test_program, 
        timeout=5,  # BigCodeBench may need more time than HumanEval
        task_id=doc["task_id"], 
        completion_id=repeat_idx
    )
    return result


def non_sanitized_predictions(resps, docs):
    """Return non-sanitized predictions for logging/debugging (complete mode)."""
    preds = []
    for resp, doc in zip(resps, docs):
        pred = [doc["complete_prompt"] + r for r in resp]
        preds.append(pred)
    return preds


def non_sanitized_predictions_instruct(resps, docs):
    """Return non-sanitized predictions for logging/debugging (instruct mode)."""
    preds = []
    for resp, doc in zip(resps, docs):
        # In instruct mode, response is the complete solution
        pred = [r for r in resp]
        preds.append(pred)
    return preds


def score_predictions(candidates: list[list[str]], docs: list[dict]):
    """Score all candidate predictions using parallel execution (complete mode).
    
    Args:
        candidates: List of candidate lists for each document
        docs: List of documents corresponding to candidates
        
    Returns:
        List of result lists maintaining the same structure as candidates
    """
    return evaluate_candidates(candidates, docs, process_resp)


def score_predictions_instruct(candidates: list[list[str]], docs: list[dict]):
    """Score all candidate predictions using parallel execution (instruct mode).
    
    Args:
        candidates: List of candidate lists for each document
        docs: List of documents corresponding to candidates
        
    Returns:
        List of result lists maintaining the same structure as candidates
    """
    return evaluate_candidates(candidates, docs, process_resp_instruct)
