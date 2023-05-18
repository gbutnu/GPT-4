import os
import time
import logging
import argparse
import json
import torch
import torch.distributed as dist

from gpt_4.model import GPT4
from gpt_4.utils import set_seed, get_rank, get_world_size
from gpt_4.data_utils import get_data_loaders

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the config file.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the data directory.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the output directory.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training.')
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Initialize distributed training
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=get_world_size(),
        rank=get_rank()
    )

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    # Load data
    train_loader, valid_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=config['batch_size'],
        local_rank=args.local_rank
    )

    # Initialize model
    model = GPT4(config)
    model.to(f'cuda:{get_rank()}')

    # Train model
    model.train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()