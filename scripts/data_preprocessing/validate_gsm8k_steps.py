#!/usr/bin/env python3
"""
Validate GSM8K dataset: Check if numbers in the first step equation appear in the question.
"""

import json
import re
from pathlib import Path
from collections import defaultdict


# Number word mappings
WORD_TO_NUM = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
    'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
    'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
    'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000, 'million': 1000000
}

ORDINAL_TO_NUM = {
    'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
    'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10,
    'eleventh': 11, 'twelfth': 12
}

FRACTION_TO_NUM = {
    'half': 0.5, 'halves': 0.5,
    'third': 1/3, 'thirds': 1/3,
    'quarter': 0.25, 'quarters': 0.25, 'fourth': 0.25, 'fourths': 0.25,
    'fifth': 0.2, 'fifths': 0.2,
    'sixth': 1/6, 'sixths': 1/6,
    'seventh': 1/7, 'sevenths': 1/7,
    'eighth': 0.125, 'eighths': 0.125,
    'ninth': 1/9, 'ninths': 1/9,
    'tenth': 0.1, 'tenths': 0.1
}


def extract_numbers_from_text(text, enhanced=False):
    """
    Extract all numbers from text, including integers and decimals.

    Args:
        text: Input text string
        enhanced: If True, also extract percentages, comma-separated numbers, text numbers, etc.

    Returns:
        Set of number strings found in the text
    """
    # Pattern to match integers and decimals (e.g., 1.50, 40, 0.2)
    # This matches numbers like: 123, 1.5, 0.25, etc.
    pattern = r'\d+\.?\d*'
    matches = re.findall(pattern, text)

    # Normalize numbers (convert to float and back to handle equivalences like 1.50 == 1.5)
    normalized = set()
    for match in matches:
        try:
            num_val = float(match)
            # Add both the original and various normalized forms
            normalized.add(match)  # Keep original
            normalized.add(str(num_val))  # Normalized float string
            if num_val == int(num_val):
                normalized.add(str(int(num_val)))  # Integer form if applicable
        except ValueError:
            normalized.add(match)

    if enhanced:
        # Extract additional number formats
        normalized.update(extract_enhanced_numbers(text))

    return normalized


def extract_enhanced_numbers(text):
    """
    Extract numbers from enhanced formats: percentages, commas, text numbers, fractions.

    Args:
        text: Input text string

    Returns:
        Set of extracted numbers
    """
    numbers = set()
    text_lower = text.lower()

    # # 1. Extract percentages: "30%" -> 30
    # percent_pattern = r'(\d+\.?\d*)\s*%'
    # for match in re.finditer(percent_pattern, text):
    #     num_str = match.group(1)
    #     num_val = float(num_str)
    #     numbers.add(str(num_val))
    #     if num_val == int(num_val):
    #         numbers.add(str(int(num_val)))

    # # 2. Extract comma-separated numbers: "1,500" -> 1500
    # comma_pattern = r'\d{1,3}(?:,\d{3})+'
    # for match in re.finditer(comma_pattern, text):
    #     num_str = match.group(0).replace(',', '')
    #     num_val = float(num_str)
    #     numbers.add(str(num_val))
    #     if num_val == int(num_val):
    #         numbers.add(str(int(num_val)))

    # 3. Extract text numbers: "two", "three", etc.
    for word, num in WORD_TO_NUM.items():
        if re.search(r'\b' + word + r'\b', text_lower):
            numbers.add(str(num))

    # 4. Extract ordinal numbers: "first", "second", etc.
    for word, num in ORDINAL_TO_NUM.items():
        if re.search(r'\b' + word + r'\b', text_lower):
            numbers.add(str(num))

    # # 5. Extract fractions: "two thirds", "one half", etc.
    # # Pattern: "two thirds" -> 2/3 = 0.666...
    # fraction_pattern = r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+(half|halves|third|thirds|quarter|quarters|fourth|fourths|fifth|fifths|sixth|sixths|seventh|sevenths|eighth|eighths|ninth|ninths|tenth|tenths)\b'
    # for match in re.finditer(fraction_pattern, text_lower):
    #     numerator_word = match.group(1)
    #     denominator_word = match.group(2)

    #     numerator = WORD_TO_NUM.get(numerator_word, 1)
    #     # Get base denominator value
    #     if denominator_word in FRACTION_TO_NUM:
    #         denominator_val = FRACTION_TO_NUM[denominator_word]
    #         result = numerator * denominator_val
    #         numbers.add(str(result))

    # # 6. Single fraction words: "half", "third", etc.
    # for word, num in FRACTION_TO_NUM.items():
    #     if re.search(r'\b' + word + r'\b', text_lower):
    #         numbers.add(str(num))

    return numbers


def extract_equation_from_step(step):
    """
    Extract equation from a step string (between << and >>).

    Args:
        step: Step string like "<<40+50=90>>"

    Returns:
        Equation string like "40+50=90", or None if not found
    """
    match = re.search(r'<<(.+?)>>', step)
    if match:
        return match.group(1)
    return None


def extract_left_side_numbers(equation):
    """
    Extract all numbers from the left side of an equation (before =).

    Args:
        equation: Equation string like "40+50=90"

    Returns:
        Set of number strings from the left side
    """
    if '=' not in equation:
        return set()

    left_side = equation.split('=')[0]
    # Extract numbers from left side
    pattern = r'\d+\.?\d*'
    numbers = re.findall(pattern, left_side)

    # Normalize numbers
    normalized = set()
    for num in numbers:
        try:
            num_val = float(num)
            normalized.add(num)
            normalized.add(str(num_val))
            if num_val == int(num_val):
                normalized.add(str(int(num_val)))
        except ValueError:
            normalized.add(num)

    return normalized


def check_numbers_in_question(left_side_numbers, question_numbers):
    """
    Check if all numbers from left side appear in question numbers.

    Args:
        left_side_numbers: Set of numbers from equation's left side
        question_numbers: Set of numbers from question text

    Returns:
        Tuple of (all_found: bool, missing_numbers: set)
    """
    # Check if each number in left side has a match in question
    missing = set()

    for left_num in left_side_numbers:
        # Check if this number or any equivalent form exists in question numbers
        found = False
        try:
            left_val = float(left_num)
            for q_num in question_numbers:
                try:
                    q_val = float(q_num)
                    if abs(left_val - q_val) < 1e-9:  # Float comparison with tolerance
                        found = True
                        break
                except ValueError:
                    continue
        except ValueError:
            if left_num in question_numbers:
                found = True

        if not found:
            missing.add(left_num)

    return len(missing) == 0, missing


def validate_sample(sample, sample_idx, use_enhanced=False):
    """
    Validate a single data sample.

    Args:
        sample: Dictionary with 'question', 'steps', 'answer'
        sample_idx: Index of the sample for error reporting
        use_enhanced: If True, use enhanced number extraction

    Returns:
        Dictionary with validation results
    """
    result = {
        'valid': False,
        'question_numbers': set(),
        'first_step': None,
        'equation': None,
        'left_side_numbers': set(),
        'missing_numbers': set(),
        'error': None,
        'enhanced_used': use_enhanced
    }

    try:
        # Extract numbers from question
        question = sample.get('question', '')
        result['question_numbers'] = extract_numbers_from_text(question, enhanced=use_enhanced)

        # Get first step
        steps = sample.get('steps', [])
        if not steps:
            result['error'] = "No steps found"
            return result

        first_step = steps[0]
        result['first_step'] = first_step

        # Extract equation from first step
        equation = extract_equation_from_step(first_step)
        if not equation:
            result['error'] = f"Could not extract equation from step: {first_step}"
            return result

        result['equation'] = equation

        # Extract numbers from left side of equation
        left_side_numbers = extract_left_side_numbers(equation)
        result['left_side_numbers'] = left_side_numbers

        if not left_side_numbers:
            result['error'] = f"No numbers found in left side of equation: {equation}"
            return result

        # Check if all left side numbers appear in question
        all_found, missing = check_numbers_in_question(
            left_side_numbers,
            result['question_numbers']
        )

        result['valid'] = all_found
        result['missing_numbers'] = missing

    except Exception as e:
        result['error'] = f"Exception: {str(e)}"

    return result


def validate_dataset(file_path, verbose=False, max_display=5, test_refinement=True):
    """
    Validate a GSM8K dataset file.

    Args:
        file_path: Path to the JSON dataset file
        verbose: If True, print details for invalid samples
        max_display: Maximum number of invalid samples to display in detail
        test_refinement: If True, test enhanced extraction on failed samples

    Returns:
        Dictionary with validation statistics
    """
    print(f"\n{'='*80}")
    print(f"Validating: {file_path}")
    print(f"{'='*80}")

    # Load dataset
    with open(file_path, 'r') as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")

    # First pass: Validate each sample with basic extraction
    stats = {
        'total': len(data),
        'valid_basic': 0,
        'invalid_basic': 0,
        'valid_enhanced': 0,
        'still_invalid': 0,
        'errors': defaultdict(int),
        'invalid_samples': [],
        'refined_samples': []
    }

    basic_invalid_samples = []

    for idx, sample in enumerate(data):
        result = validate_sample(sample, idx, use_enhanced=False)

        if result['valid']:
            stats['valid_basic'] += 1
        else:
            stats['invalid_basic'] += 1
            if result['error']:
                stats['errors'][result['error']] += 1
            basic_invalid_samples.append((idx, sample, result))

    # Second pass: Try enhanced extraction on failed samples
    if test_refinement and basic_invalid_samples:
        print(f"\nTesting enhanced extraction on {len(basic_invalid_samples)} failed samples...")

        for idx, sample, basic_result in basic_invalid_samples:
            # Skip samples with errors (not missing numbers)
            if basic_result['error']:
                stats['still_invalid'] += 1
                stats['invalid_samples'].append((idx, sample, basic_result))
                continue

            # Try enhanced extraction
            enhanced_result = validate_sample(sample, idx, use_enhanced=True)

            if enhanced_result['valid']:
                stats['valid_enhanced'] += 1
                stats['refined_samples'].append((idx, sample, basic_result, enhanced_result))
            else:
                stats['still_invalid'] += 1
                stats['invalid_samples'].append((idx, sample, enhanced_result))
    else:
        stats['still_invalid'] = stats['invalid_basic']
        stats['invalid_samples'] = basic_invalid_samples

    # Print statistics
    print(f"\nResults (Basic Extraction):")
    print(f"  ✓ Valid samples:   {stats['valid_basic']} ({stats['valid_basic']/stats['total']*100:.2f}%)")
    print(f"  ✗ Invalid samples: {stats['invalid_basic']} ({stats['invalid_basic']/stats['total']*100:.2f}%)")

    if test_refinement:
        print(f"\nRefinement Results:")
        print(f"  ✓ Fixed by enhanced extraction: {stats['valid_enhanced']} ({stats['valid_enhanced']/stats['total']*100:.2f}%)")
        print(f"  ✗ Still invalid:                {stats['still_invalid']} ({stats['still_invalid']/stats['total']*100:.2f}%)")
        total_valid = stats['valid_basic'] + stats['valid_enhanced']
        print(f"\n  Total valid (with refinement): {total_valid} ({total_valid/stats['total']*100:.2f}%)")
        improvement = stats['valid_enhanced'] / stats['invalid_basic'] * 100 if stats['invalid_basic'] > 0 else 0
        print(f"  Improvement: {improvement:.2f}% of failed samples recovered")

    if stats['errors']:
        print(f"\nError breakdown:")
        for error, count in sorted(stats['errors'].items(), key=lambda x: -x[1]):
            print(f"  - \"{error}\" (count: {count})")

    # Display refined samples (those fixed by enhancement)
    if verbose and test_refinement and stats['refined_samples']:
        print(f"\n{'='*80}")
        print(f"Samples fixed by enhanced extraction (showing first {max_display}):")
        print(f"{'='*80}")

        for idx, sample, basic_result, enhanced_result in stats['refined_samples'][:max_display]:
            print(f"\nSample #{idx}:")
            print(f"  Question: {sample['question'][:150]}...")
            print(f"  First step: {basic_result['first_step']}")
            print(f"  Equation: {basic_result['equation']}")
            print(f"  Basic question numbers: {sorted(basic_result['question_numbers'], key=lambda x: float(x) if x.replace('.','').replace('-','').isdigit() else 0)}")
            print(f"  Enhanced question numbers: {sorted(enhanced_result['question_numbers'], key=lambda x: float(x) if x.replace('.','').replace('-','').isdigit() else 0)}")
            print(f"  Left side numbers: {sorted(basic_result['left_side_numbers'], key=lambda x: float(x) if x.replace('.','').replace('-','').isdigit() else 0)}")
            print(f"  ✓ Was missing: {basic_result['missing_numbers']} -> NOW FOUND")

    # Display still invalid samples
    if verbose and stats['invalid_samples']:
        print(f"\n{'='*80}")
        print(f"Still invalid sample details (showing first {max_display}):")
        print(f"{'='*80}")

        for idx, sample, result in stats['invalid_samples'][:max_display]:
            print(f"\nSample #{idx}:")
            print(f"  Question: {sample['question'][:150]}...")
            print(f"  First step: {result['first_step']}")
            print(f"  Equation: {result['equation']}")
            print(f"  Question numbers: {sorted(result['question_numbers'], key=lambda x: float(x) if x.replace('.','').replace('-','').isdigit() else 0)}")
            print(f"  Left side numbers: {sorted(result['left_side_numbers'], key=lambda x: float(x) if x.replace('.','').replace('-','').isdigit() else 0)}")
            if result['missing_numbers']:
                print(f"  ✗ Missing numbers: {result['missing_numbers']}")
            if result['error']:
                print(f"  ✗ Error: {result['error']}")

    return stats


def main():
    """Main function to validate all GSM8K dataset files."""

    # Dataset paths
    base_path = Path("/home/yinx/yinx/recursive-internal/data/input/datasets/gsm")
    splits = ['train', 'test', 'valid']

    all_stats = {}

    for split in splits:
        file_path = base_path / f"{split}.json"

        if not file_path.exists():
            print(f"\nWarning: {file_path} not found, skipping...")
            continue

        # Validate dataset (set verbose=True to see invalid sample details)
        stats = validate_dataset(file_path, verbose=True, max_display=5, test_refinement=True)
        all_stats[split] = stats

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    print(f"\n{'Split':<10s} {'Basic Valid':<15s} {'Enhanced Valid':<20s} {'Total Valid':<15s} {'Improvement':<15s}")
    print(f"{'-'*10} {'-'*15} {'-'*20} {'-'*15} {'-'*15}")

    for split, stats in all_stats.items():
        basic_valid = stats['valid_basic']
        enhanced_valid = stats['valid_enhanced']
        total_valid = basic_valid + enhanced_valid
        total = stats['total']

        basic_pct = basic_valid / total * 100 if total > 0 else 0
        total_pct = total_valid / total * 100 if total > 0 else 0
        improvement_pct = enhanced_valid / stats['invalid_basic'] * 100 if stats['invalid_basic'] > 0 else 0

        print(f"{split:<10s} {basic_valid:6d}/{total:6d} ({basic_pct:5.2f}%)  "
              f"{enhanced_valid:6d} recovered     "
              f"{total_valid:6d}/{total:6d} ({total_pct:5.2f}%)  "
              f"{improvement_pct:5.2f}% recovered")

    print()


if __name__ == "__main__":
    main()
