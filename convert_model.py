import argparse
import torch

from openai.gpt4.model import GPT4Model


def main(args):
    # Load the GPT-4 model
    model = GPT4Model.from_file(args.model_file)

    # Convert the model to PyTorch
    torch_model = model.to_torch()

    # Save the PyTorch model
    torch.save(torch_model, args.output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', help='Path to the GPT-4 model file')
    parser.add_argument('output_file', help='Path to the output PyTorch model file')
    args = parser.parse_args()
    main(args)