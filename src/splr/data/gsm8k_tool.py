# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
import argparse
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any
from tqdm import tqdm


def normalize_number(num_str: str) -> str:
    """
    Normalize a number string by:
    1. Adding leading 0 for decimals like .5 -> 0.5
    2. Removing trailing zeros after decimal: 4.50 -> 4.5, 4.00 -> 4
    3. Removing commas: 1,500 -> 1500

    Args:
        num_str: Number string to normalize

    Returns:
        Normalized number string
    """
    # Remove commas
    num_str = num_str.replace(',', '')

    # Add leading 0 for decimals like .5
    if num_str.startswith('.'):
        num_str = '0' + num_str

    # Remove trailing zeros after decimal point
    if '.' in num_str:
        # Convert to float and back to remove trailing zeros
        try:
            num_val = float(num_str)
            # Check if it's a whole number
            if num_val == int(num_val):
                return str(int(num_val))
            else:
                # Remove trailing zeros but keep at least one decimal place if needed
                normalized = f"{num_val:.10f}".rstrip('0').rstrip('.')
                return normalized
        except ValueError:
            return num_str

    return num_str


def normalize_text(text: str) -> str:
    """
    Normalize all numbers in a text string.

    Args:
        text: Input text

    Returns:
        Text with normalized numbers
    """
    # Pattern to match numbers (with optional commas and decimals)
    # Matches: 1,500  .5  4.50  123  etc.
    pattern = r'\d{1,3}(?:,\d{3})+(?:\.\d+)?|\.\d+|\d+\.\d+|\d+'

    def replace_number(match):
        return normalize_number(match.group(0))

    return re.sub(pattern, replace_number, text)


def parse_and_split_equation(equation: str) -> Tuple[List[str], bool, str]:
    """
    Parse an equation and split multi-operator expressions into single-operator steps.

    Args:
        equation: Equation string like "30+25+35=90" or "2+3*4=14"

    Returns:
        Tuple of:
        - List of intermediate step equations
        - Boolean indicating if division was not exact
        - Error message if any
    """
    if '=' not in equation:
        return [], False, "No equals sign found"

    left_side, right_side = equation.split('=', 1)
    left_side = left_side.strip()
    right_side = right_side.strip()

    try:
        expected_result = float(right_side)
    except ValueError:
        return [], False, f"Invalid right side: {right_side}"

    # Tokenize the left side
    tokens = tokenize_expression(left_side)
    if not tokens:
        return [], False, "Failed to tokenize expression"

    # Evaluate and generate steps
    steps, inexact_division = evaluate_with_steps(tokens, expected_result)

    return steps, inexact_division, ""


def tokenize_expression(expr: str) -> List[Tuple[str, str]]:
    """
    Tokenize a mathematical expression into (type, value) pairs.

    Args:
        expr: Expression string like "30+25*2"

    Returns:
        List of (token_type, token_value) tuples
        token_type can be: 'number', 'operator', 'lparen', 'rparen'
    """
    tokens = []
    i = 0
    expr = expr.replace(' ', '')

    while i < len(expr):
        # Check for numbers (including decimals and negative numbers)
        if expr[i].isdigit() or expr[i] == '.':
            j = i
            while j < len(expr) and (expr[j].isdigit() or expr[j] == '.'):
                j += 1
            tokens.append(('number', expr[i:j]))
            i = j
        # Check for negative numbers at start or after operators/parentheses
        elif expr[i] == '-' and (i == 0 or expr[i-1] in '(+-*/'):
            # This is a negative number
            j = i + 1
            while j < len(expr) and (expr[j].isdigit() or expr[j] == '.'):
                j += 1
            if j > i + 1:  # We found digits after the minus
                tokens.append(('number', expr[i:j]))
                i = j
            else:
                # Just a minus operator
                tokens.append(('operator', expr[i]))
                i += 1
        # Check for operators
        elif expr[i] in '+-*/':
            tokens.append(('operator', expr[i]))
            i += 1
        # Check for parentheses
        elif expr[i] == '(':
            tokens.append(('lparen', '('))
            i += 1
        elif expr[i] == ')':
            tokens.append(('rparen', ')'))
            i += 1
        else:
            # Unknown character, skip
            i += 1

    return tokens


def evaluate_with_steps(tokens: List[Tuple[str, str]], expected_result: float) -> Tuple[List[str], bool]:
    """
    Evaluate expression and generate intermediate steps.
    Follows standard order of operations: parentheses, then *, /, then +, -

    Args:
        tokens: List of (token_type, token_value) tuples
        expected_result: Expected final result for validation

    Returns:
        Tuple of (list of step equations, inexact_division flag)
    """
    steps = []
    inexact_division = False

    # If only one number, no steps needed
    if len(tokens) == 1 and tokens[0][0] == 'number':
        return [], False

    # Process the expression step by step
    # First handle parentheses, then * and /, then + and -

    # For simplicity, we'll use a different approach:
    # Convert to postfix and evaluate, tracking intermediate results

    # Actually, let's use a simpler approach for common cases
    # Check if expression has only one type of operator and no parentheses

    operators = [t for t in tokens if t[0] == 'operator']
    has_parens = any(t[0] in ['lparen', 'rparen'] for t in tokens)

    if not has_parens and len(set(op[1] for op in operators)) == 1:
        # All operators are the same type, process left to right
        return process_uniform_operators(tokens, expected_result)

    # For mixed operators, we need to respect precedence
    return process_with_precedence(tokens, expected_result)


def process_uniform_operators(tokens: List[Tuple[str, str]], expected_result: float) -> Tuple[List[str], bool]:
    """
    Process expression with uniform operators (all +, or all *, etc.)

    Args:
        tokens: List of tokens
        expected_result: Expected final result

    Returns:
        Tuple of (list of steps, inexact_division flag)
    """
    steps = []
    inexact_division = False

    # Extract numbers and operators
    numbers = [float(t[1]) for t in tokens if t[0] == 'number']
    operators = [t[1] for t in tokens if t[0] == 'operator']

    if len(numbers) < 2:
        return [], False

    # Process left to right
    result = numbers[0]

    for i, op in enumerate(operators):
        next_num = numbers[i + 1]

        # Generate step equation
        left_expr = f"{result}{op}{next_num}" if '.' not in str(result) or result == int(result) else f"{result:.10f}".rstrip('0').rstrip('.') + op + str(next_num)

        # Calculate result
        if op == '+':
            new_result = result + next_num
        elif op == '-':
            new_result = result - next_num
        elif op == '*':
            new_result = result * next_num
        elif op == '/':
            if next_num == 0:
                return [], False  # Division by zero
            new_result = result / next_num
            # Check if division is exact
            if new_result != int(new_result) and (result / next_num) * next_num != result:
                inexact_division = True
        else:
            new_result = result

        # Format the result
        result_str = str(int(new_result)) if new_result == int(new_result) else f"{new_result:.10f}".rstrip('0').rstrip('.')

        # Clean up left expression
        left_expr = normalize_expression_numbers(left_expr)

        step_eq = f"<<{left_expr}={result_str}>>"
        steps.append(step_eq)

        result = new_result

    return steps, inexact_division


def process_with_precedence(tokens: List[Tuple[str, str]], expected_result: float) -> Tuple[List[str], bool]:
    """
    Process expression respecting operator precedence.

    Args:
        tokens: List of tokens
        expected_result: Expected final result

    Returns:
        Tuple of (list of steps, inexact_division flag)
    """
    steps = []
    inexact_division = False

    # Create a working copy of tokens
    working_tokens = tokens.copy()

    # First, handle parentheses (innermost first)
    while any(t[0] == 'lparen' for t in working_tokens):
        # Find innermost parentheses
        lparen_idx = -1
        for i, t in enumerate(working_tokens):
            if t[0] == 'lparen':
                lparen_idx = i

        if lparen_idx == -1:
            break

        # Find matching rparen
        rparen_idx = -1
        for i in range(lparen_idx + 1, len(working_tokens)):
            if working_tokens[i][0] == 'rparen':
                rparen_idx = i
                break

        if rparen_idx == -1:
            return [], False  # Mismatched parentheses

        # Extract tokens inside parentheses
        inner_tokens = working_tokens[lparen_idx + 1:rparen_idx]

        # Evaluate inner expression
        inner_steps, inner_inexact = process_simple_expression(inner_tokens)
        steps.extend(inner_steps)
        inexact_division = inexact_division or inner_inexact

        # Get final result of inner expression
        if inner_steps:
            last_step = inner_steps[-1]
            # Extract result from last step: <<...=result>>
            result_match = re.search(r'=([0-9.]+)>>', last_step)
            if result_match:
                inner_result = result_match.group(1)
            else:
                inner_result = str(eval_tokens(inner_tokens))
        else:
            inner_result = str(eval_tokens(inner_tokens))

        # Replace parentheses group with result
        working_tokens = working_tokens[:lparen_idx] + [('number', inner_result)] + working_tokens[rparen_idx + 1:]

    # Now handle multiplication and division
    while any(t[0] == 'operator' and t[1] in '*/' for t in working_tokens):
        # Find first * or /
        for i, t in enumerate(working_tokens):
            if t[0] == 'operator' and t[1] in '*/':
                # Get operands
                if i == 0 or i >= len(working_tokens) - 1:
                    return [], False

                # Validate that we have numbers on both sides
                if working_tokens[i - 1][0] != 'number' or working_tokens[i + 1][0] != 'number':
                    return [], False

                try:
                    left_num = float(working_tokens[i - 1][1])
                    op = t[1]
                    right_num = float(working_tokens[i + 1][1])
                except (ValueError, IndexError):
                    return [], False

                # Calculate
                if op == '*':
                    result = left_num * right_num
                else:  # op == '/'
                    if right_num == 0:
                        return [], False
                    result = left_num / right_num
                    if result != int(result):
                        inexact_division = True

                # Format result
                result_str = str(int(result)) if result == int(result) else f"{result:.10f}".rstrip('0').rstrip('.')

                # Create step
                left_str = str(int(left_num)) if left_num == int(left_num) else f"{left_num:.10f}".rstrip('0').rstrip('.')
                right_str = str(int(right_num)) if right_num == int(right_num) else f"{right_num:.10f}".rstrip('0').rstrip('.')

                step_eq = f"<<{left_str}{op}{right_str}={result_str}>>"
                steps.append(step_eq)

                # Replace in tokens
                working_tokens = working_tokens[:i - 1] + [('number', result_str)] + working_tokens[i + 2:]
                break

    # Finally handle addition and subtraction (left to right)
    while any(t[0] == 'operator' and t[1] in '+-' for t in working_tokens):
        # Find first + or -
        for i, t in enumerate(working_tokens):
            if t[0] == 'operator' and t[1] in '+-':
                # Get operands
                if i == 0 or i >= len(working_tokens) - 1:
                    return [], False

                # Validate that we have numbers on both sides
                if working_tokens[i - 1][0] != 'number' or working_tokens[i + 1][0] != 'number':
                    return [], False

                try:
                    left_num = float(working_tokens[i - 1][1])
                    op = t[1]
                    right_num = float(working_tokens[i + 1][1])
                except (ValueError, IndexError):
                    return [], False

                # Calculate
                if op == '+':
                    result = left_num + right_num
                else:  # op == '-'
                    result = left_num - right_num

                # Format result
                result_str = str(int(result)) if result == int(result) else f"{result:.10f}".rstrip('0').rstrip('.')

                # Create step
                left_str = str(int(left_num)) if left_num == int(left_num) else f"{left_num:.10f}".rstrip('0').rstrip('.')
                right_str = str(int(right_num)) if right_num == int(right_num) else f"{right_num:.10f}".rstrip('0').rstrip('.')

                step_eq = f"<<{left_str}{op}{right_str}={result_str}>>"
                steps.append(step_eq)

                # Replace in tokens
                working_tokens = working_tokens[:i - 1] + [('number', result_str)] + working_tokens[i + 2:]
                break

    return steps, inexact_division


def process_simple_expression(tokens: List[Tuple[str, str]]) -> Tuple[List[str], bool]:
    """
    Process a simple expression without parentheses.
    """
    # Use the precedence-aware processor
    return process_with_precedence(tokens, 0)


def eval_tokens(tokens: List[Tuple[str, str]]) -> float:
    """
    Evaluate tokens to get a numeric result.
    """
    expr_str = ''.join(t[1] for t in tokens)
    try:
        return eval(expr_str)
    except:
        return 0


def normalize_expression_numbers(expr: str) -> str:
    """
    Normalize numbers within an expression string.
    """
    pattern = r'\d+\.\d+'

    def replace_num(match):
        num_str = match.group(0)
        num_val = float(num_str)
        if num_val == int(num_val):
            return str(int(num_val))
        return f"{num_val:.10f}".rstrip('0').rstrip('.')

    return re.sub(pattern, replace_num, expr)


def has_operator_in_equation(equation: str) -> bool:
    """
    Check if an equation has any operators in the left side.

    Args:
        equation: Equation string like "8=8" or "2+3=5"

    Returns:
        True if there are operators, False otherwise
    """
    if '=' not in equation:
        return False

    left_side = equation.split('=')[0].strip()

    # Check if any operator exists in the left side
    return any(op in left_side for op in ['+', '-', '*', '/', '(', ')'])


def process_sample_multi_normal(sample: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, bool, str]:
    """
    Process a single sample to generate multi-step normal form.

    Args:
        sample: Dictionary with 'question', 'steps', 'answer'

    Returns:
        Tuple of (processed_sample, should_skip, inexact_division_flag, error_message)
        should_skip: True if sample should be excluded from output
    """
    try:
        # Check if steps is empty - skip if so
        if not sample.get('steps') or len(sample.get('steps', [])) == 0:
            return sample, True, False, "Empty steps"

        # Normalize question
        normalized_question = normalize_text(sample['question'])

        # Process each step and check for trivial equations
        new_steps = []
        inexact_division = False
        has_trivial_equation = False

        for step in sample['steps']:
            # Normalize the step
            normalized_step = normalize_text(step)

            # Extract equation from step
            match = re.search(r'<<(.+?)>>', normalized_step)
            if not match:
                # Keep step as is if no equation found
                new_steps.append(normalized_step)
                continue

            equation = match.group(1)

            # Check if equation has operators in left side
            if not has_operator_in_equation(equation):
                # This is a trivial equation like <<8=8>>
                has_trivial_equation = True
                # Still process the rest to see if there are other steps
                continue

            # Split multi-operator equation into single-operator steps
            try:
                split_steps, step_inexact, error = parse_and_split_equation(equation)

                if error:
                    # If error, keep original step
                    new_steps.append(normalized_step)
                elif split_steps:
                    # Add all split steps
                    new_steps.extend(split_steps)
                    inexact_division = inexact_division or step_inexact
                else:
                    # Single operator or already simple, keep normalized version
                    new_steps.append(normalized_step)
            except Exception as e:
                # On any exception, keep the original normalized step
                new_steps.append(normalized_step)

        # Skip sample if it has trivial equations
        if has_trivial_equation:
            return sample, True, False, "Contains trivial equation (no operators)"

        # Skip if no valid steps after processing
        if len(new_steps) == 0:
            return sample, True, False, "No valid steps after processing"
        
        # TODO: We currently skip the samples with too many steps (>12) since they're not appearing in the test/valid split
        if len(new_steps) > 12:
            return sample, True, False, "Too many steps"

        # Normalize answer
        normalized_answer = normalize_text(sample['answer'])

        processed = {
            'question': normalized_question,
            'steps': new_steps,
            'answer': normalized_answer
        }

        return processed, False, inexact_division, ""

    except Exception as e:
        # Return original sample and skip on error
        return sample, True, False, str(e)


def validate_steps(steps: List[str], answer: str) -> Tuple[bool, str]:
    """
    Validate that steps follow correct format and final answer matches.

    Args:
        steps: List of step strings like ["<<2+3=5>>", "<<5*4=20>>"]
        answer: Expected final answer

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not steps:
        return False, "No steps provided"

    for i, step in enumerate(steps):
        # Extract equation from step
        match = re.search(r'<<(.+?)>>', step)
        if not match:
            return False, f"Step {i}: No equation found"

        equation = match.group(1)

        if '=' not in equation:
            return False, f"Step {i}: No equals sign in equation"

        parts = equation.split('=', 1)
        left_side = parts[0].strip()
        right_side = parts[1].strip() if len(parts) > 1 else ""

        # Check if right side is empty
        if not right_side:
            return False, f"Step {i}: Empty right side"

        # Check if left side has proper format: number operator number
        # Tokenize the left side to validate structure
        tokens = tokenize_expression(left_side)

        if not tokens:
            return False, f"Step {i}: Failed to tokenize left side"

        # Count numbers and operators
        numbers_count = sum(1 for t in tokens if t[0] == 'number')
        operators_count = sum(1 for t in tokens if t[0] == 'operator')

        # Must have at least 2 numbers and 1 operator
        if numbers_count < 2:
            return False, f"Step {i}: Left side missing numbers (found {numbers_count}, need at least 2)"
        if operators_count < 1:
            return False, f"Step {i}: Left side missing operator"

        # Additional check: ensure operators are surrounded by numbers or parentheses
        for j, token in enumerate(tokens):
            if token[0] == 'operator':
                # Check if there's something before and after
                if j == 0:
                    return False, f"Step {i}: Operator at start of expression"
                if j == len(tokens) - 1:
                    return False, f"Step {i}: Operator at end of expression"
                # Check neighbors - should be number or paren
                if tokens[j-1][0] not in ['number', 'rparen']:
                    return False, f"Step {i}: No number before operator"
                if tokens[j+1][0] not in ['number', 'lparen']:
                    return False, f"Step {i}: No number after operator"

        # If this is the final step, check if right side matches answer
        if i == len(steps) - 1:
            try:
                # Normalize both values for comparison
                right_normalized = normalize_number(right_side)
                answer_normalized = normalize_number(answer)

                # Compare as floats to handle different representations
                try:
                    right_val = float(right_normalized)
                    answer_val = float(answer_normalized)
                    if abs(right_val - answer_val) > 1e-9:
                        return False, f"Final step result ({right_normalized}) does not match answer ({answer_normalized})"
                except ValueError:
                    # Fall back to string comparison
                    if right_normalized != answer_normalized:
                        return False, f"Final step result ({right_normalized}) does not match answer ({answer_normalized})"
            except Exception as e:
                return False, f"Final step: Error comparing result and answer: {e}"

    return True, ""


def extract_operators_from_steps(steps: List[str]) -> List[str]:
    """
    Extract the list of operators used in the steps in order.

    Args:
        steps: List of step strings like ["<<2+3=5>>", "<<5*4=20>>"]

    Returns:
        List of operator symbols like ['+', '*']
    """
    operators = []

    for step in steps:
        # Extract equation from step
        match = re.search(r'<<(.+?)>>', step)
        if not match:
            continue

        equation = match.group(1)

        if '=' not in equation:
            continue

        left_side = equation.split('=')[0]

        # Find the operator in the left side
        # Look for the operators in order
        for op in ['+', '-', '*', '/']:
            if op in left_side:
                operators.append(op)
                break  # Only one operator per step

    return operators


def load_metadata(output_dir: Path) -> Dict[str, Any]:
    """
    Load metadata.json if it exists, otherwise create a new one.

    Args:
        output_dir: Directory where metadata.json is stored

    Returns:
        Metadata dictionary
    """
    metadata_path = output_dir / "metadata.json"

    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    else:
        # Create new metadata with task ID 0 for empty list
        return {
            "summary": {
                "max_steps_length": {},
                "steps_length_distribution": {},
                "total_tasks": 1
            },
            "tasks": {
                "0": []
            },
            "task_to_id": {
                "[]": 0
            }
        }


def save_metadata(metadata: Dict[str, Any], output_dir: Path):
    """
    Save metadata.json.

    Args:
        metadata: Metadata dictionary
        output_dir: Directory where metadata.json is stored
    """
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def get_task_id(operators: List[str], metadata: Dict[str, Any], split: str, allow_new: bool = True) -> int:
    """
    Get or create a task ID for a given operator sequence.

    Args:
        operators: List of operators like ['+', '*', '+']
        metadata: Metadata dictionary
        split: Dataset split (train, test, valid)
        allow_new: If True (for train), create new IDs. If False (for test/valid), return 0 for unknown tasks.

    Returns:
        Task ID (integer)
    """
    # Create string key for the operator list
    task_key = json.dumps(operators)

    # Check if task already exists
    if task_key in metadata["task_to_id"]:
        return metadata["task_to_id"][task_key]

    # If not found and we're not allowed to create new (test/valid)
    if not allow_new:
        return 0

    # Create new task ID (for train split)
    new_id = metadata["summary"]["total_tasks"]
    metadata["summary"]["total_tasks"] = new_id + 1
    metadata["tasks"][str(new_id)] = operators
    metadata["task_to_id"][task_key] = new_id

    return new_id


def generate_multi_normal_data(data: List[Dict[str, Any]], split: str) -> Dict[str, Any]:
    """
    Generate multi-normal version of the dataset.

    Args:
        data: List of data samples
        split: Dataset split name (train, test, valid)

    Returns:
        Dictionary with statistics
    """
    # Create output directory
    output_dir = Path("./datasets/gsm8k/tool_normalized")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata = load_metadata(output_dir)

    # Determine if we can add new task IDs (only for train)
    allow_new_tasks = (split == 'train')

    processed_data = []
    stats = {
        'total': len(data),
        'processed': 0,
        'skipped': 0,
        'skipped_empty_steps': 0,
        'skipped_trivial_equations': 0,
        'skipped_no_valid_steps': 0,
        'skipped_errors': 0,
        'skipped_invalid_format': 0,  # New: invalid step format
        'skipped_answer_mismatch': 0,  # New: final answer doesn't match
        'inexact_division_count': 0,
        'max_steps_length': 0,
        'unknown_tasks_count': 0,  # For test/valid samples with tasks not in metadata
        'steps_length_distribution': {},  # Track count for each step length
    }

    if not allow_new_tasks:
        print(f"Note: This is {split} split. New task patterns will be assigned task_id=0")

    for i, sample in enumerate(tqdm(data, desc=f"Processing {split}", unit="sample")):
        processed_sample, should_skip, inexact_division, error = process_sample_multi_normal(sample)

        if should_skip:
            stats['skipped'] += 1
            # Track specific skip reasons
            if error:
                if "Empty steps" in error:
                    stats['skipped_empty_steps'] += 1
                elif "trivial equation" in error:
                    stats['skipped_trivial_equations'] += 1
                elif "No valid steps" in error or 'Too many steps' in error:
                    stats['skipped_no_valid_steps'] += 1
                else:
                    stats['skipped_errors'] += 1
            continue  # Don't add to processed_data

        if inexact_division:
            stats['inexact_division_count'] += 1

        # Validate steps format and answer match
        is_valid, validation_error = validate_steps(processed_sample['steps'], processed_sample['answer'])
        if not is_valid:
            stats['skipped'] += 1
            # Categorize the validation error
            if "answer" in validation_error.lower() or "match" in validation_error.lower():
                stats['skipped_answer_mismatch'] += 1
            else:
                stats['skipped_invalid_format'] += 1
            continue  # Don't add to processed_data

        # Extract operators from steps
        operators = extract_operators_from_steps(processed_sample['steps'])

        # Get task ID
        task_id = get_task_id(operators, metadata, split, allow_new=allow_new_tasks)

        # Track if this is an unknown task for test/valid
        if task_id == 0 and len(operators) > 0 and not allow_new_tasks:
            stats['unknown_tasks_count'] += 1
            print(f"  Warning: Unknown task pattern in {split} sample {i}: {operators}")
        
        # if task_id == 0:
        #     # TODO: Currently we skip the unseen task patterns / no valid steps
        #     stats['skipped_no_valid_steps'] += 1
        #     stats['skipped'] += 1
        #     continue

        # Add task and task_id to sample
        processed_sample['task'] = operators
        processed_sample['task_id'] = task_id

        # Track max steps length
        steps_length = len(processed_sample['steps'])
        if steps_length > stats['max_steps_length']:
            stats['max_steps_length'] = steps_length

        # Track steps length distribution
        steps_length_key = str(steps_length)
        if steps_length_key not in stats['steps_length_distribution']:
            stats['steps_length_distribution'][steps_length_key] = 0
        stats['steps_length_distribution'][steps_length_key] += 1

        processed_data.append(processed_sample)
        stats['processed'] += 1

    # Update metadata summary for this split
    metadata['summary']['max_steps_length'][split] = stats['max_steps_length']
    metadata['summary']['steps_length_distribution'][split] = stats['steps_length_distribution']

    # Save metadata
    save_metadata(metadata, output_dir)

    # Save processed data
    output_path = output_dir / f"{split}.json"
    with open(output_path, 'w') as f:
        json.dump(processed_data, f, indent=2)

    print(f"\n=== Multi-Normal Processing Statistics ===")
    print(f"Total samples: {stats['total']}")
    print(f"Processed samples: {stats['processed']} ({stats['processed']/stats['total']*100:.2f}%)")
    print(f"Skipped samples: {stats['skipped']} ({stats['skipped']/stats['total']*100:.2f}%)")
    print(f"  - Empty steps: {stats['skipped_empty_steps']}")
    print(f"  - Trivial equations (no operators): {stats['skipped_trivial_equations']}")
    print(f"  - No valid steps after processing: {stats['skipped_no_valid_steps']}")
    print(f"  - Invalid step format: {stats['skipped_invalid_format']}")
    print(f"  - Answer mismatch: {stats['skipped_answer_mismatch']}")
    print(f"  - Other errors: {stats['skipped_errors']}")
    print(f"Samples with inexact division: {stats['inexact_division_count']}")
    print(f"Max steps length: {stats['max_steps_length']}")
    if not allow_new_tasks and stats['unknown_tasks_count'] > 0:
        print(f"Unknown task patterns (assigned task_id=0): {stats['unknown_tasks_count']}")
    print(f"Total unique tasks in metadata: {metadata['summary']['total_tasks']}")

    # Print step length distribution
    print(f"\nSteps length distribution:")
    if stats['steps_length_distribution']:
        # Sort by step length (as integers)
        sorted_lengths = sorted(stats['steps_length_distribution'].items(),
                               key=lambda x: int(x[0]))
        for length, count in sorted_lengths:
            percentage = count / stats['processed'] * 100 if stats['processed'] > 0 else 0
            print(f"  {length} step(s): {count:6d} samples ({percentage:5.2f}%)")

    print(f"\nSaved to: {output_path}")

    return stats


def generate_data(args):
    """
    Convert icot text data to JSON format.
    Args:
        split (str): The dataset split (e.g., train, test, valid).
    """
    from transformers import AutoTokenizer

    # Load tokenizer for token counting
    print("Loading tokenizer for token counting...")
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    with open(f"./datasets/gsm8k/{args.split}.txt") as f:
        data = f.readlines()

    # Initialize token statistics
    token_stats = {
        "question": {"samples": [], "total": 0},
        "steps": {"samples": [], "total": 0},
        "answer": {"samples": [], "total": 0},
        "all": {"samples": [], "total": 0}
    }

    data = [
        {
            "question": d.split("||")[0],
            "steps": d.split("||")[1].split("##")[0].strip().split(" "),
            "answer": d.split("##")[-1].strip(),
        }
        for d in data
    ]

    # Count tokens for each sample
    print(f"Counting tokens for {len(data)} samples...")
    for sample in data:
        question = sample["question"]
        steps = " ".join(sample["steps"])
        answer = sample["answer"]
        all_text = f"{question} {steps} {answer}"

        question_tokens = len(tokenizer.encode(question))
        steps_tokens = len(tokenizer.encode(steps))
        answer_tokens = len(tokenizer.encode(answer))
        all_tokens = len(tokenizer.encode(all_text))

        token_stats["question"]["samples"].append(question_tokens)
        token_stats["question"]["total"] += question_tokens

        token_stats["steps"]["samples"].append(steps_tokens)
        token_stats["steps"]["total"] += steps_tokens

        token_stats["answer"]["samples"].append(answer_tokens)
        token_stats["answer"]["total"] += answer_tokens

        token_stats["all"]["samples"].append(all_tokens)
        token_stats["all"]["total"] += all_tokens

    # Print statistics
    print("\n=== Token Statistics ===")
    for field_name, stats in token_stats.items():
        samples = stats["samples"]
        if not samples:
            continue

        min_count = min(samples)
        max_count = max(samples)
        avg_count = stats["total"] // len(samples)

        print(f"\n{field_name.upper()}:")
        print(f"  Min: {min_count} tokens")
        print(f"  Max: {max_count} tokens")
        print(f"  Average: {avg_count} tokens")
        print(f"  Total: {stats['total']} tokens")

    json.dump(data, open(f"./datasets/gsm8k/{args.split}.json", "w"))

    # Generate multi-normal version
    print(f"\n{'='*80}")
    print("Generating multi-normal version...")
    print(f"{'='*80}")
    generate_multi_normal_data(data, args.split)


def load_dataset_accum(file_paths, tokenizer, config):
    """
    Load and preprocess GSM8K dataset for multi-step reasoning with accumulated context.
    Each turn's input includes the full conversation history (previous inputs + outputs).
    Labels include full sequence with input text masked out.

    Args:
        file_paths: List of paths to JSON dataset files (or single path string)
        tokenizer: Tokenizer instance for encoding text
        config: Model configuration containing max_position_embeddings, max_reasoning_steps, task_emb_ndim, task_emb_len, hidden_size

    Returns:
        List of processed samples with 'inputs', 'labels', and 'task_identifiers' tensors
        Each sample has:
        - inputs: (max_reasoning_steps, max_position_embeddings) tensor
        - labels: (max_reasoning_steps, max_position_embeddings) tensor with input portion masked
        - task_identifiers: (max_reasoning_steps,) tensor
    """
    import torch
    from pathlib import Path
    import re

    # Ensure file_paths is a list
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    print(f"Loading GSM8K multi-step ACCUMULATED dataset from {len(file_paths)} file(s)")

    # Get max_reasoning_steps from config
    max_reasoning_steps = getattr(config, 'max_reasoning_steps', 12)

    # Calculate task embedding length
    task_emb_ndim = getattr(config, 'task_emb_ndim', 0)
    task_emb_len_cfg = getattr(config, 'task_emb_len', 0)
    num_task_identifiers = getattr(config, 'num_task_identifiers', 0)

    if task_emb_ndim > 0:
        # Calculate task_emb_len using ceiling division
        task_emb_len = -(task_emb_ndim // -config.hidden_size) if task_emb_len_cfg == 0 else task_emb_len_cfg
        input_max_len = config.max_position_embeddings - task_emb_len
        print(f"Task embeddings enabled: task_emb_len={task_emb_len}, input_max_len={input_max_len}")
    else:
        task_emb_len = 0
        input_max_len = config.max_position_embeddings
        print(f"Task embeddings disabled: input_max_len={input_max_len}")

    print(f"Max reasoning steps: {max_reasoning_steps}")

    # Load data from all files
    all_data = []
    for file_path in file_paths:
        print(f"  Loading {file_path}...")
        with open(file_path, 'r') as f:
            data = json.load(f)
            all_data.extend(data)

    print(f"Loaded {len(all_data)} total samples")

    # Process each sample
    processed = []
    skipped_count = 0

    for i, sample in enumerate(tqdm(all_data, desc="Tokenizing samples")):
        try:
            question = sample["question"]
            steps = sample.get("steps", [])
            task_id = sample.get("task_id", 0) if num_task_identifiers > 1 else 0

            # Skip if no steps
            if not steps or len(steps) == 0:
                skipped_count += 1
                continue

            # Initialize lists for all turns
            inputs_list = []
            labels_list = []
            task_ids_list = []

            # Track accumulated conversation history
            accumulated_text = question

            # Process each step
            for step_idx, step in enumerate(steps):
                if step_idx >= max_reasoning_steps:
                    break

                # Extract equation from step: <<equation_before=equation_after>>
                match = re.search(r"<<(.+?)=(.+?)>>", step)
                if not match:
                    print(f"Warning: Could not extract equation from step '{step}' in sample {i}")
                    raise ValueError(f"Invalid step format: {step}")

                equation_before = match.group(1)  # Left side of equation (input to operation)
                equation_after = match.group(2)   # Right side of equation (result)

                # Build input for this turn
                if step_idx == 0:
                    # First turn: just the question
                    input_text = question
                else:
                    # Subsequent turns: accumulated_text already contains previous conversation
                    input_text = accumulated_text

                # Tokenize input
                input_tokens = tokenizer.encode(
                    input_text,
                    max_length=input_max_len,
                    truncation=True,
                    padding="max_length"
                )

                # Tokenize input text alone to know how many tokens to mask
                input_text_tokens = tokenizer.encode(
                    input_text,
                    add_special_tokens=False,
                    truncation=False
                )

                if step_idx == len(steps) - 1:
                    label_text = input_text + tokenizer.eos_token + f"<tool_call>{equation_before}</tool_call>"
                else:
                    label_text = f"{input_text}<tool_call>{equation_before}</tool_call>"

                # Tokenize label with right padding
                label_tokens = tokenizer.encode(
                    label_text,
                    add_special_tokens=False,
                    padding="max_length",
                    max_length=input_max_len,
                )

                # if len(label_tokens) > config.max_position_embeddings:
                #     print(f"Warning: Label tokens length {len(label_tokens)} exceeds max_position_embeddings {config.max_position_embeddings} in sample {i}")

                # Convert to tensor and mask out input tokens
                label_tokens_tensor = torch.tensor(label_tokens, dtype=torch.long)
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

                # Mask the first len(input_text_tokens) tokens
                input_token_len = len(input_text_tokens)
                for j in range(min(input_token_len, len(label_tokens))):
                    label_tokens_tensor[j] = pad_token_id
                
                if config.max_position_embeddings > input_max_len:
                    label_tokens_tensor = torch.cat([torch.full((config.max_position_embeddings - input_max_len,), pad_token_id, dtype=torch.long), label_tokens_tensor])

                # Add to lists
                inputs_list.append(torch.tensor(input_tokens, dtype=torch.long))
                labels_list.append(label_tokens_tensor)
                task_ids_list.append(torch.tensor(task_id, dtype=torch.long))

                # Update accumulated text for next turn
                # Accumulate: previous conversation + tool_call + tool_response
                accumulated_text = f"{input_text}<tool_call>{equation_before}</tool_call><tool_response>{equation_after}</tool_response>"

                # if step_idx == len(steps) - 1 and step_idx + 1 < max_reasoning_steps:
                #     input_text = accumulated_text

                #     # Tokenize input
                #     input_tokens = tokenizer.encode(
                #         input_text,
                #         max_length=input_max_len,
                #         truncation=True,
                #         padding="max_length"
                #     )

                #     # Tokenize input text alone to know how many tokens to mask
                #     input_text_tokens = tokenizer.encode(
                #         input_text,
                #         add_special_tokens=False,
                #         truncation=False
                #     )

                #     label_text = input_text + tokenizer.eos_token

                #     # Tokenize label with right padding
                #     label_tokens = tokenizer.encode(
                #         label_text,
                #         add_special_tokens=False,
                #         padding="max_length",
                #         max_length=input_max_len,
                #     )

                #     # if len(label_tokens) > config.max_position_embeddings:
                #     #     print(f"Warning: Label tokens length {len(label_tokens)} exceeds max_position_embeddings {config.max_position_embeddings} in sample {i}")

                #     # Convert to tensor and mask out input tokens
                #     label_tokens_tensor = torch.tensor(label_tokens, dtype=torch.long)
                #     pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

                #     # Mask the first len(input_text_tokens) tokens
                #     input_token_len = len(input_text_tokens)
                #     for j in range(min(input_token_len, len(label_tokens))):
                #         label_tokens_tensor[j] = pad_token_id
                    
                #     if config.max_position_embeddings > input_max_len:
                #         label_tokens_tensor = torch.cat([torch.full((config.max_position_embeddings - input_max_len,), pad_token_id, dtype=torch.long), label_tokens_tensor])

                #     # Add to lists
                #     inputs_list.append(torch.tensor(input_tokens, dtype=torch.long))
                #     labels_list.append(label_tokens_tensor)
                #     task_ids_list.append(torch.tensor(task_id, dtype=torch.long))


            # Pad remaining turns with pad tokens
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            pad_input = torch.full((input_max_len,), pad_token_id, dtype=torch.long)
            pad_label = torch.full((config.max_position_embeddings,), pad_token_id, dtype=torch.long)
            pad_task_id = torch.tensor(-1, dtype=torch.long)

            for _ in range(len(inputs_list), max_reasoning_steps):
                inputs_list.append(pad_input)
                labels_list.append(pad_label)
                task_ids_list.append(pad_task_id)

            # Stack into tensors
            inputs_tensor = torch.stack(inputs_list)  # (max_reasoning_steps, input_max_len)
            labels_tensor = torch.stack(labels_list)  # (max_reasoning_steps, max_position_embeddings)
            task_ids_tensor = torch.stack(task_ids_list)  # (max_reasoning_steps,)

            processed.append({
                "inputs": inputs_tensor,
                "labels": labels_tensor,
                "task_identifiers": task_ids_tensor
            })

        except Exception as e:
            print(f"Warning: Error processing sample {i}: {e}")
            skipped_count += 1
            continue

    print(f"Successfully processed {len(processed)} samples from GSM8K multi-step ACCUMULATED dataset")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} samples due to errors or missing data")

    return processed


def load_dataset(file_paths, tokenizer, config):
    """
    Load and preprocess GSM8K dataset for multi-step reasoning.
    Labels include full sequence with input text masked out.

    Args:
        file_paths: List of paths to JSON dataset files (or single path string)
        tokenizer: Tokenizer instance for encoding text
        config: Model configuration containing max_position_embeddings, max_reasoning_steps, task_emb_ndim, task_emb_len, hidden_size

    Returns:
        List of processed samples with 'inputs', 'labels', and 'task_identifiers' tensors
        Each sample has:
        - inputs: (max_reasoning_steps, max_position_embeddings) tensor
        - labels: (max_reasoning_steps, max_position_embeddings) tensor with input portion masked
        - task_identifiers: (max_reasoning_steps,) tensor
    """
    import torch
    from pathlib import Path
    import re

    # Ensure file_paths is a list
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    print(f"Loading GSM8K multi-step dataset from {len(file_paths)} file(s)")

    # Get max_reasoning_steps from config
    max_reasoning_steps = getattr(config, 'max_reasoning_steps', 12)

    # Calculate task embedding length
    task_emb_ndim = getattr(config, 'task_emb_ndim', 0)
    task_emb_len_cfg = getattr(config, 'task_emb_len', 0)
    num_task_identifiers = getattr(config, 'num_task_identifiers', 0)

    if task_emb_ndim > 0:
        # Calculate task_emb_len using ceiling division
        task_emb_len = -(task_emb_ndim // -config.hidden_size) if task_emb_len_cfg == 0 else task_emb_len_cfg
        input_max_len = config.max_position_embeddings - task_emb_len
        print(f"Task embeddings enabled: task_emb_len={task_emb_len}, input_max_len={input_max_len}")
    else:
        task_emb_len = 0
        input_max_len = config.max_position_embeddings
        print(f"Task embeddings disabled: input_max_len={input_max_len}")

    print(f"Max reasoning steps: {max_reasoning_steps}")

    # Load data from all files
    all_data = []
    for file_path in file_paths:
        print(f"  Loading {file_path}...")
        with open(file_path, 'r') as f:
            data = json.load(f)
            all_data.extend(data)

    print(f"Loaded {len(all_data)} total samples")

    # Process each sample
    processed = []
    skipped_count = 0

    for i, sample in enumerate(tqdm(all_data, desc="Tokenizing samples")):
        try:
            question = sample["question"]
            steps = sample.get("steps", [])
            task_id = sample.get("task_id", 0) if num_task_identifiers > 1 else 0

            # Skip if no steps
            if not steps or len(steps) == 0:
                skipped_count += 1
                continue

            # Initialize lists for all turns
            inputs_list = []
            labels_list = []
            task_ids_list = []

            # Track previous result for building input
            prev_call = None
            prev_result = None

            # Process each step
            for step_idx, step in enumerate(steps):
                if step_idx >= max_reasoning_steps:
                    break

                # Extract equation from step: <<equation_before=equation_after>>
                match = re.search(r"<<(.+?)=(.+?)>>", step)
                if not match:
                    print(f"Warning: Could not extract equation from step '{step}' in sample {i}")
                    raise ValueError(f"Invalid step format: {step}")

                equation_before = match.group(1)  # Left side of equation (input to operation)
                equation_after = match.group(2)   # Right side of equation (result)

                # Build input for this turn
                if step_idx == 0:
                    # First turn: just the question
                    input_text = question
                else:
                    # Subsequent turns: question + <tool_response> + previous result + </tool_response>
                    input_text = f"{question}<tool_call>{prev_call}</tool_call><tool_response>{prev_result}</tool_response>"

                # Tokenize input
                input_tokens = tokenizer.encode(
                    input_text,
                    max_length=input_max_len,
                    truncation=True,
                    padding="max_length"
                )

                # Tokenize input text alone to know how many tokens to mask
                input_text_tokens = tokenizer.encode(
                    input_text,
                    add_special_tokens=False,
                    truncation=False
                )

                # Build label text
                if step_idx == len(steps) - 1:
                    label_text = input_text + tokenizer.eos_token + f"<tool_call>{equation_before}</tool_call><tool_response>{equation_after}</tool_response>"
                else:
                    label_text = f"{input_text}<tool_call>{equation_before}</tool_call>"

                # Tokenize label with right padding
                label_tokens = tokenizer.encode(
                    label_text,
                    add_special_tokens=False,
                    padding="max_length",
                    max_length=input_max_len,
                )

                # Convert to tensor and mask out input tokens
                label_tokens_tensor = torch.tensor(label_tokens, dtype=torch.long)
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

                # Mask the first len(input_text_tokens) tokens
                input_token_len = len(input_text_tokens)
                for j in range(min(input_token_len, len(label_tokens))):
                    label_tokens_tensor[j] = pad_token_id

                if config.max_position_embeddings > input_max_len:
                    label_tokens_tensor = torch.cat([torch.full((config.max_position_embeddings - input_max_len,), pad_token_id, dtype=torch.long), label_tokens_tensor])

                # Add to lists
                inputs_list.append(torch.tensor(input_tokens, dtype=torch.long))
                labels_list.append(label_tokens_tensor)
                task_ids_list.append(torch.tensor(task_id, dtype=torch.long))

                # Save result for next turn
                prev_result = equation_after
                prev_call = equation_before

                # if step_idx == len(steps) - 1 and step_idx + 1 < max_reasoning_steps:
                #     # Build input for this turn
                #     if step_idx == 0:
                #         # First turn: just the question
                #         input_text = question
                #     else:
                #         # Subsequent turns: question + <tool_response> + previous result + </tool_response>
                #         input_text = f"{question}<tool_call>{prev_call}</tool_call><tool_response>{prev_result}</tool_response>"

                #     # Tokenize input
                #     input_tokens = tokenizer.encode(
                #         input_text,
                #         max_length=input_max_len,
                #         truncation=True,
                #         padding="max_length"
                #     )

                #     # Tokenize input text alone to know how many tokens to mask
                #     input_text_tokens = tokenizer.encode(
                #         input_text,
                #         add_special_tokens=False,
                #         truncation=False
                #     )

                #     label_text = input_text + tokenizer.eos_token

                #     # Tokenize label with right padding
                #     label_tokens = tokenizer.encode(
                #         label_text,
                #         add_special_tokens=False,
                #         padding="max_length",
                #         max_length=input_max_len,
                #     )

                #     # Convert to tensor and mask out input tokens
                #     label_tokens_tensor = torch.tensor(label_tokens, dtype=torch.long)
                #     pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

                #     # Mask the first len(input_text_tokens) tokens
                #     input_token_len = len(input_text_tokens)
                #     for j in range(min(input_token_len, len(label_tokens))):
                #         label_tokens_tensor[j] = pad_token_id

                #     if config.max_position_embeddings > input_max_len:
                #         label_tokens_tensor = torch.cat([torch.full((config.max_position_embeddings - input_max_len,), pad_token_id, dtype=torch.long), label_tokens_tensor])

                #     # Add to lists
                #     inputs_list.append(torch.tensor(input_tokens, dtype=torch.long))
                #     labels_list.append(label_tokens_tensor)
                #     task_ids_list.append(torch.tensor(task_id, dtype=torch.long))

            # Pad remaining turns with pad tokens
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            pad_input = torch.full((input_max_len,), pad_token_id, dtype=torch.long)
            pad_label = torch.full((config.max_position_embeddings,), pad_token_id, dtype=torch.long)
            pad_task_id = torch.tensor(-1, dtype=torch.long)

            for _ in range(len(inputs_list), max_reasoning_steps):
                inputs_list.append(pad_input)
                labels_list.append(pad_label)
                task_ids_list.append(pad_task_id)

            # Stack into tensors
            inputs_tensor = torch.stack(inputs_list)  # (max_reasoning_steps, input_max_len)
            labels_tensor = torch.stack(labels_list)  # (max_reasoning_steps, max_position_embeddings)
            task_ids_tensor = torch.stack(task_ids_list)  # (max_reasoning_steps,)

            processed.append({
                "inputs": inputs_tensor,
                "labels": labels_tensor,
                "task_identifiers": task_ids_tensor
            })

        except Exception as e:
            print(f"Warning: Error processing sample {i}: {e}")
            skipped_count += 1
            continue

    print(f"Successfully processed {len(processed)} samples from GSM8K multi-step dataset")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} samples due to errors or missing data")

    return processed


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="Convert icot text data to JSON format."
    # )
    # parser.add_argument(
    #     "split", type=str, help="The dataset split (e.g., train, test, valid)."
    # )
    # args = parser.parse_args()
    # generate_data(args)

    from transformers import AutoTokenizer

    class Config:
        max_position_embeddings = 256
        hidden_size = 1024
        task_emb_ndim = 1024  # Set to > 0 to test task embeddings
        task_emb_len = 1
        max_reasoning_steps = 6  # For multi-step dataset

    # Load tokenizer and add chat template
    tokenizer_path = ""
    # tokenizer_path = "Qwen/Qwen3-0.6B"

    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Test with multi_normal dataset
    file_paths = [
        "./datasets/gsm8k/tool_normalized/test.json"
    ]

    # print(f"\n{'='*80}")
    # print("Testing load_dataset (single-step)")
    # print(f"{'='*80}")
    # processed = load_dataset(file_paths, tokenizer, Config)

    # if processed:
    #     print(f"\n=== Sample 0 ===")
    #     print(f"Inputs shape: {processed[0]['inputs'].shape}")
    #     print(f"Labels shape: {processed[0]['labels'].shape}")
    #     print(f"Task ID: {processed[0]['task_identifiers'].item()}")

    #     print(f"\nDecoded inputs:")
    #     print(tokenizer.decode(processed[0]["inputs"]))

    #     print(f"\nDecoded labels:")
    #     print(tokenizer.decode(processed[0]["labels"]))

    #     # Show a few more samples
    #     print(f"\n=== Sample 1 ===")
    #     print(f"Task ID: {processed[1]['task_identifiers'].item()}")
    #     print(f"Decoded inputs: {tokenizer.decode(processed[1]['inputs'])}")
    #     print(f"Decoded labels: {tokenizer.decode(processed[1]['labels'])}")
    # else:
    #     print("No samples processed")

    # Test with multi_normal dataset (multi-step)
    print(f"\n{'='*80}")
    print("Testing load_dataset_multi (multi-step)")
    print(f"{'='*80}")
    processed_multi = load_dataset_accum(file_paths, tokenizer, Config)
    # processed_multi = load_dataset(file_paths, tokenizer, Config)

    if processed_multi:
        print(f"\n=== Multi-step Sample 0 ===")
        print(f"Inputs shape: {processed_multi[0]['inputs'].shape}")  # Should be (max_reasoning_steps, max_position_embeddings)
        print(f"Labels shape: {processed_multi[0]['labels'].shape}")  # Should be (max_reasoning_steps, max_position_embeddings)
        print(f"Task IDs shape: {processed_multi[0]['task_identifiers'].shape}")  # Should be (max_reasoning_steps,)

        # Show each turn
        for turn_idx in range(Config.max_reasoning_steps):
            task_id = processed_multi[0]['task_identifiers'][turn_idx].item()
            if task_id == -1:
                print(f"\nTurn {turn_idx}: [PADDING]")
                break
            else:
                print(f"\nTurn {turn_idx} (Task ID: {task_id}):")
                print(f"  Input: {tokenizer.decode(processed_multi[0]['inputs'][turn_idx])}")
                print(f"  Label: {tokenizer.decode(processed_multi[0]['labels'][turn_idx])}")

        print(f"\n=== Multi-step Sample 1 ===")
        print(f"Number of actual steps: {(processed_multi[1]['task_identifiers'] != -1).sum().item()}")
        for turn_idx in range(min(4, Config.max_reasoning_steps)):  # Show first 4 turns
            task_id = processed_multi[1]['task_identifiers'][turn_idx].item()
            if task_id == -1:
                print(f"\nTurn {turn_idx}: [PADDING]")
                break
            else:
                print(f"\nTurn {turn_idx} (Task ID: {task_id}):")
                print(f"  Input: {tokenizer.decode(processed_multi[1]['inputs'][turn_idx])}")
                print(f"  Label: {tokenizer.decode(processed_multi[1]['labels'][turn_idx])}")
    else:
        print("No multi-step samples processed")