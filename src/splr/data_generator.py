import argparse


def main():
    parser = argparse.ArgumentParser(description="Dataset statistics generator and data processor")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="Dataset name (e.g., gsm8k)")
    # parser.add_argument("--data_dir", type=str, default="", help="Data directory for the dataset")
    parser.add_argument("--split", type=str, default="train", help="Split of the dataset")
    args = parser.parse_args()

    generate_dataset_data(args)


def generate_dataset_data(args):
    """
    Generate/process data for the specified dataset.

    Args:
        args: Command line arguments
    """
    dataset_name = args.dataset.lower()

    # Dynamically import the appropriate dataset module
    if dataset_name == "gsm8k-tool":
        from splr.data import gsm8k_tool
        gsm8k_tool.generate_data(args)
    elif dataset_name == "gsm8k":
        from splr.data import gsm8k
        gsm8k.generate_data(args)
    else:
        print(f"Error: Dataset '{dataset_name}' is not supported yet")


if __name__ == "__main__":
    main()