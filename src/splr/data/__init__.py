from typing import List, Dict
import torch


def load_dataset(dataset: str, file_paths: List[str], tokenizer, config, input_mode: str = "recurrent") -> List[Dict[str, torch.Tensor]]:
    """
    Generic dataset loader that dispatches to specific dataset implementations.

    Args:
        dataset_name: Name of the dataset
        file_paths: List of paths to dataset files (or single path as a list)
        tokenizer: Tokenizer instance for encoding text
        config: Model configuration
        input_mode: "recurrent" or "autoregressive" — controls how multi-step input is built

    Returns:
        List of processed samples with 'inputs' and 'labels' tensors

    Raises:
        ValueError: If dataset_name is not recognized
    """
    if dataset == "gsm8k-tool":
        from . import gsm8k_tool
        if input_mode == "autoregressive":
            return gsm8k_tool.load_dataset_accum(file_paths, tokenizer, config)
        else:
            return gsm8k_tool.load_dataset(file_paths, tokenizer, config)
    elif dataset == "gsm8k":
        from . import gsm8k
        if input_mode == "autoregressive":
            return gsm8k.load_dataset_accum(file_paths, tokenizer, config)
        else:
            return gsm8k.load_dataset(file_paths, tokenizer, config)
    # Add more dataset loaders here as needed
    else:
        raise ValueError(f"Unknown dataset name: {dataset}. Supported datasets: ['gsm8k', 'gsm8k-tool']")



__all__ = ['load_dataset']
