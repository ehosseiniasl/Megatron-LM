"""Post-processing LLM-generated Python code implemented using tree-sitter."""

import os
import pathlib
from typing import Dict, Generator, List, Optional, Set, Tuple
from collections import deque

import tree_sitter_python
from tqdm import tqdm
from tree_sitter import Language, Node, Parser

from evalplus.data import (
    get_human_eval_plus,
    get_mbpp_plus,
    load_solutions,
    write_directory,
    write_jsonl,
)
from evalplus.syncheck import syntax_check

CLASS_TYPE = "class_definition"
FUNCTION_TYPE = "function_definition"
IMPORT_TYPE = ["import_statement", "import_from_statement"]
IDENTIFIER_TYPE = "identifier"
ATTRIBUTE_TYPE = "attribute"
RETURN_TYPE = "return_statement"
EXPRESSION_TYPE = "expression_statement"
ASSIGNMENT_TYPE = "assignment"


def code_extract(text: str) -> str:
    lines = text.split("\n")
    n = len(lines)

    longest_range = (0, 0)
    longest_so_far = 0

    for i in range(n):
        current_length = 0
        for j in range(i, n):
            if lines[j].strip():  # Count only non-empty lines
                current_length += 1

            if current_length <= longest_so_far:
                continue  # Skip if it's already shorter than the best

            current_lines = "\n".join(lines[i : j + 1])
            if syntax_check(current_lines):
                longest_range = (i, j)
                longest_so_far = current_length  # Update longest valid block

    return "\n".join(lines[longest_range[0] : longest_range[1] + 1])


def get_deps_old(nodes: List[Tuple[str, Node]]) -> Dict[str, Set[str]]:
    def dfs_get_deps(node: Node, deps: Set[str]) -> None:
        for child in node.children:
            if child.type == IDENTIFIER_TYPE:
                deps.add(child.text.decode("utf8"))
            else:
                dfs_get_deps(child, deps)

    name2deps = {}
    for name, node in nodes:
        deps = set()
        dfs_get_deps(node, deps)
        name2deps[name] = deps
    return name2deps


def get_deps(nodes: List[Tuple[str, Node]]) -> Dict[str, Set[str]]:
    name2deps = {}

    for name, root in nodes:
        deps = set()
        stack = [root]

        while stack:
            node = stack.pop()
            if node.type == IDENTIFIER_TYPE:
                deps.add(node.text.decode("utf8"))
            else:
                stack.extend(node.children)  # Process children without recursion

        name2deps[name] = deps

    return name2deps


def get_function_dependency(
    entrypoint: str, call_graph: Dict[str, Set[str]]
) -> Set[str]:
    queue = deque([entrypoint])  # Use deque for O(1) popleft()
    visited = {entrypoint}

    while queue:
        current = queue.popleft()  # Faster than pop(0)
        if current in call_graph:
            for neighbor in call_graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)  # O(1) append to the right

    return visited


def get_definition_name(node: Node) -> str:
    for child in node.children:
        if child.type == IDENTIFIER_TYPE:
            return child.text.decode("utf8")


def traverse_tree(node: Node) -> Generator[Node, None, None]:
    cursor = node.walk()
    depth = 0

    visited_children = False
    while True:
        if not visited_children:
            yield cursor.node
            if not cursor.goto_first_child():
                depth += 1
                visited_children = True
        elif cursor.goto_next_sibling():
            visited_children = False
        elif not cursor.goto_parent() or depth == 0:
            break
        else:
            depth -= 1


def has_return_statement(node: Node) -> bool:
    traverse_nodes = traverse_tree(node)
    for node in traverse_nodes:
        if node.type == RETURN_TYPE:
            return True
    return False


def extract_target_code_or_empty(code: str, entrypoint: Optional[str] = None) -> str:
    code = code_extract(code)
    code_bytes = code.encode("utf8")  # Use encode instead of bytes()
    parser = Parser(Language(tree_sitter_python.language()))
    tree = parser.parse(code_bytes)

    root_node = tree.root_node
    import_nodes = []
    definition_nodes = {}

    # Collect class, function, and variable definitions
    for child in root_node.children:
        node_type = child.type
        if node_type in IMPORT_TYPE:
            import_nodes.append(child)
        elif node_type in {CLASS_TYPE, FUNCTION_TYPE}:
            name = get_definition_name(child)
            if name and name not in definition_nodes:
                if node_type == FUNCTION_TYPE and not has_return_statement(child):
                    continue  # Skip functions without return statements
                definition_nodes[name] = child
        elif node_type == EXPRESSION_TYPE and child.children[0].type == ASSIGNMENT_TYPE:
            name = get_definition_name(child.children[0])
            if name and name not in definition_nodes:
                definition_nodes[name] = child.children[0]

    # If entrypoint is provided, filter by dependencies
    if entrypoint:
        name2deps = get_deps(list(definition_nodes.items()))
        reachable = get_function_dependency(entrypoint, name2deps)
        definition_nodes = {
            name: node for name, node in definition_nodes.items() if name in reachable
        }

    # Build sanitized output
    sanitized_output = bytearray()

    for node in import_nodes:
        sanitized_output.extend(code_bytes[node.start_byte : node.end_byte] + b"\n")

    for node in definition_nodes.values():
        sanitized_output.extend(code_bytes[node.start_byte : node.end_byte] + b"\n")

    return sanitized_output.rstrip(b"\n").decode("utf8")


def sanitize(code: str, entrypoint: Optional[str] = None) -> str:
    sanitized_code = extract_target_code_or_empty(code, entrypoint).strip()
    if not sanitized_code:
        return code_extract(code)
    return sanitized_code


def script(
    samples: str, inplace: bool = False, debug_task: str = None, mbpp_version="default"
):
    # task_id -> entry_point
    entry_point = {}
    # merge two datasets
    dataset = {**get_human_eval_plus(), **get_mbpp_plus(version=mbpp_version)}

    for task_id, problem in dataset.items():
        entry_point[task_id] = problem["entry_point"]

    # make a new folder with "-sanitized" suffix
    is_folder = os.path.isdir(samples)
    target_path = pathlib.Path(samples)
    if not inplace:
        if is_folder:
            new_name = target_path.name + "-sanitized"
        else:
            new_name = target_path.name.replace(".jsonl", "-sanitized.jsonl")
        target_path = target_path.parent / new_name
    target_path = str(target_path)

    nsan = 0
    ntotal = 0

    new_solutions = []

    for solution in tqdm(load_solutions(samples)):
        task_id = solution["task_id"]
        if task_id not in dataset:
            print(
                f"Skiping {task_id} as it does not existing in the latest EvalPlus dataset."
            )
            continue

        function_name = entry_point[task_id] if task_id in entry_point else None
        dbg_identifier = solution["_identifier"]
        if debug_task is not None and task_id != debug_task:
            continue

        ntotal += 1
        if "solution" in solution:
            old_code = solution["solution"]
        else:
            assert "completion" in solution
            old_code = dataset[task_id]["prompt"] + "\n" + solution["completion"]

        new_code = sanitize(code=old_code, entrypoint=function_name)

        # if changed, print the message
        if new_code != old_code:
            msg = "Sanitized: " + dbg_identifier
            if is_folder:
                msg += " -> " + dbg_identifier.replace(samples, target_path)
            print(msg)
            nsan += 1

        new_solutions.append({"task_id": task_id, "solution": new_code})

    if is_folder:
        write_directory(target_path, new_solutions)
    else:
        write_jsonl(target_path, new_solutions)

    if nsan > 0:
        print(f"Sanitized {nsan} out of {ntotal} files.")
    else:
        print(f"All files seems valid -- no files are sanitized.")
    print(f"Check the sanitized files at {target_path}")


def main():
    from fire import Fire

    Fire(script)


if __name__ == "__main__":
    main()
