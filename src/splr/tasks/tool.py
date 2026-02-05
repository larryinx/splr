"""Tool task processing for SPLR evaluation.

Handles extraction and computation of tool calls in the format:
    <tool_call>formula</tool_call>
    <tool_response>result</tool_response>
"""

import re
from typing import List, Tuple, Optional


def validate_and_compute_formula(formula: str) -> Tuple[Optional[float], str]:
    """
    Validate and compute a mathematical formula string.

    Supports basic arithmetic: +, -, *, / with parentheses.
    Uses the same tokenizer/evaluator logic as gsm8k_tool.py.

    Args:
        formula: Mathematical expression string like "2+3*4"

    Returns:
        Tuple of (result, error_message).
        result is None if computation failed.
    """
    formula = formula.strip()
    if not formula:
        return None, "Empty formula"

    # Remove any = and everything after (handle "2+3=5" format)
    if '=' in formula:
        formula = formula.split('=')[0].strip()

    # Sanitize: only allow digits, operators, parens, dots, spaces, minus
    if not re.match(r'^[\d+\-*/().  ]+$', formula):
        return None, f"Invalid characters in formula: {formula}"

    # Reject exponentiation (**) — can hang eval() on adversarial inputs
    if '**' in formula:
        return None, "Exponentiation not supported"

    try:
        # Use Python eval with restricted globals for safety
        result = eval(formula, {"__builtins__": {}}, {})
        if isinstance(result, (int, float)):
            return float(result), ""
        return None, f"Non-numeric result: {result}"
    except Exception as e:
        return None, f"Computation error: {e}"


def compare_answers(predicted: Optional[str], expected: str, tolerance: float = 1e-6) -> bool:
    """
    Compare predicted and expected answers numerically.

    Args:
        predicted: Predicted answer string (may be None)
        expected: Expected answer string
        tolerance: Numerical comparison tolerance

    Returns:
        True if answers match within tolerance
    """
    if predicted is None:
        return False

    try:
        # Clean up strings
        pred_str = str(predicted).strip().replace(',', '')
        exp_str = str(expected).strip().replace(',', '')

        pred_val = float(pred_str)
        exp_val = float(exp_str)

        return abs(pred_val - exp_val) <= tolerance
    except (ValueError, TypeError):
        return False


def extract_last_tool_response(text: str) -> Optional[str]:
    """
    Find the last <tool_response>...</tool_response> content in text.

    Args:
        text: Text potentially containing tool_response tags

    Returns:
        Content of the last tool_response, or None if not found
    """
    matches = re.findall(r'<tool_response>(.*?)</tool_response>', text)
    if matches:
        return matches[-1].strip()
    return None


def postprocess_tool_output(
    output_ids: "torch.Tensor",
    input_lengths: List[int],
    tokenizer,
) -> List[Tuple[Optional[str], Optional[float], bool]]:
    """
    Process model output tokens to extract tool calls and compute results.

    For each sample in the batch:
    1. Decode output tokens after the input portion
    2. Look for <tool_call>...</tool_call> pattern
    3. If found: extract formula, compute result
    4. If EOS found first (or no tool_call): mark as done

    Args:
        output_ids: (batch, seq_len) tensor of output token IDs
        input_lengths: List of actual input lengths per sample
        tokenizer: Tokenizer for decoding

    Returns:
        List of (formula_text, computed_result, is_done) per sample.
        formula_text: the extracted formula string, or None
        computed_result: the float result if formula was valid, or None
        is_done: True if the model produced EOS (no more tool calls)
    """
    import torch

    batch_size = output_ids.shape[0]
    results = []

    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    tool_call_id = tokenizer.encode("<tool_call>", add_special_tokens=False)[0]
    tool_call_end_id = tokenizer.encode("</tool_call>", add_special_tokens=False)[0]

    for i in range(batch_size):
        try:
            # Get the output portion (after input)
            input_len = input_lengths[i]
            output_tokens = output_ids[i, input_len:]
            has_eos = False
            if output_tokens[0] == eos_token_id:
                has_eos = True

            tool_call_start_idx = None
            tool_call_end_idx = None
            for i, token in enumerate(output_tokens):
                if token == tool_call_id:
                    tool_call_start_idx = i
                elif token == tool_call_end_id:
                    tool_call_end_idx = i
                    break
            
            if tool_call_start_idx is None or tool_call_end_idx is None:
                results.append((None, None, has_eos))
                continue
            
            output_tokens = output_tokens[tool_call_start_idx+1:tool_call_end_idx]
            decoded = tokenizer.decode(output_tokens, skip_special_tokens=False)

            result, error = validate_and_compute_formula(decoded)
            results.append((decoded, result, has_eos))
        except Exception as e:
            results.append((None, None, True))
            print(f"Error processing sample {i}: {e}")

    return results
