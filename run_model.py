import argparse
import os
import sys

import torch

from openai.openai.gpt4.model import GPT, GPTConfig
from openai.openai.gpt4.utils import load_weight_from_checkpoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, type=str, help="Name of the model")
    parser.add_argument("--model_config", required=True, type=str, help="Path to the model config file")
    parser.add_argument("--checkpoint_path", required=True, type=str, help="Path to the model checkpoint")
    parser.add_argument("--output_dir", required=True, type=str, help="Path to the output directory")
    args = parser.parse_args()

    # Load the model config
    config = GPTConfig.from_json_file(args.model_config)

    # Create the model
    model = GPT(config)

    # Load the weights from the checkpoint
    load_weight_from_checkpoint(model, args.checkpoint_path)

    # Save the model
    output_model_file = os.path.join(args.output_dir, args.model_name + ".pth")
    torch.save(model.state_dict(), output_model_file)

if __name__ == "__main__":
    main()