"""Think task processing for SPLR evaluation.

Handles extraction and computation of think steps in the format:
    <think>equation</think>

where equation is e.g. "2+3=5". The step content includes both the
formula and result. We extract the result from after the = sign.

Training format (from gsm8k.py):
    recurrent input:  question<think>prev_equation</think>
    autoregressive input:  accumulated...<think>prev_equation</think>
    label:  input<think>equation</think>  (non-final)
            input + eos + <think>equation</think>  (final)
"""

import re
from typing import List, Tuple, Optional


def validate_and_extract_think(content: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Validate and extract formula/result from a think step content.

    The content is expected to be an equation like "2+3=5" or just
    a number. We extract the result (right side of =).

    Args:
        content: Content inside <think>...</think> tags, e.g. "2+3=5"

    Returns:
        Tuple of (full_content, result_str).
        full_content: the raw content string
        result_str: the result portion (after = if present, else the whole thing)
    """
    content = content.strip()
    if not content:
        return None, None

    if '=' in content:
        result_str = content.split('=')[-1].strip()
    else:
        result_str = content

    return content, result_str


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
        pred_str = str(predicted).strip().replace(',', '')
        exp_str = str(expected).strip().replace(',', '')

        pred_val = float(pred_str)
        exp_val = float(exp_str)

        return abs(pred_val - exp_val) <= tolerance
    except (ValueError, TypeError):
        return False


def extract_last_think_response(text: str) -> Optional[str]:
    """
    Find the last <think>...</think> content in text and extract its result.

    Args:
        text: Text potentially containing think tags

    Returns:
        The result portion of the last think tag, or None if not found
    """
    matches = re.findall(r'<think>(.*?)</think>', text)
    if matches:
        content = matches[-1].strip()
        _, result_str = validate_and_extract_think(content)
        return result_str
    return None


def postprocess_think_output(
    output_ids: "torch.Tensor",
    input_lengths: List[int],
    tokenizer,
) -> List[Tuple[Optional[str], Optional[str], bool]]:
    """
    Process model output tokens to extract think steps.

    For each sample in the batch:
    1. Look at output tokens after the input portion
    2. Check if first token is EOS → done
    3. Look for <think>...</think> pattern
    4. If found: extract full content and result

    Args:
        output_ids: (batch, seq_len) tensor of output token IDs
        input_lengths: List of actual input lengths per sample
        tokenizer: Tokenizer for decoding

    Returns:
        List of (think_content, result_str, is_done) per sample.
        think_content: the full content inside <think> tags, or None
        result_str: the extracted result (after = if present), or None
        is_done: True if the model produced EOS (no more steps)
    """
    import torch

    batch_size = output_ids.shape[0]
    results = []

    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    think_open_id = tokenizer.encode("<think>", add_special_tokens=False)[0]
    think_close_id = tokenizer.encode("</think>", add_special_tokens=False)[0]

    for i in range(batch_size):
        input_len = input_lengths[i]
        output_tokens = output_ids[i, input_len:]

        has_eos = False
        if output_tokens[0] == eos_token_id:
            has_eos = True

        think_start_idx = None
        think_end_idx = None
        for j, token in enumerate(output_tokens):
            if token == think_open_id:
                think_start_idx = j
            elif token == think_close_id:
                think_end_idx = j
                break

        if think_start_idx is None or think_end_idx is None:
            results.append((None, None, has_eos))
            continue

        content_tokens = output_tokens[think_start_idx + 1:think_end_idx]
        decoded = tokenizer.decode(content_tokens, skip_special_tokens=False)

        think_content, result_str = validate_and_extract_think(decoded)
        results.append((think_content, result_str, has_eos))

    return results
