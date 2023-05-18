"""This script evaluates a GPT-4 model on a given dataset.

Usage:
    eval.py [options] <model_dir> <data_dir>

Options:
    --batch_size BATCH_SIZE  Batch size [default: 32]
    --max_seq_len MAX_SEQ_LEN  Maximum sequence length [default: 128]
    --num_eval_batches NUM_EVAL_BATCHES  Number of evaluation batches [default: 10]
    --num_gpus NUM_GPUS  Number of GPUs to use [default: 1]
    --num_samples NUM_SAMPLES  Number of samples to generate [default: 10]
    --seed SEED  Random seed [default: 42]
    --verbose  Enable verbose logging

"""

import os
import logging
import random

import numpy as np
import torch

from gpt4.model import GPT4
from gpt4.data import GPT4DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(model_dir, data_dir, batch_size=32, max_seq_len=128,
             num_eval_batches=10, num_gpus=1, num_samples=10, seed=42,
             verbose=False):
    """Evaluate a GPT-4 model.

    Args:
        model_dir (str): Path to the model directory.
        data_dir (str): Path to the data directory.
        batch_size (int): Batch size.
        max_seq_len (int): Maximum sequence length.
        num_eval_batches (int): Number of evaluation batches.
        num_gpus (int): Number of GPUs to use.
        num_samples (int): Number of samples to generate.
        seed (int): Random seed.
        verbose (bool): Enable verbose logging.

    Returns:
        float: Average perplexity.
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if num_gpus > 0:
        torch.cuda.manual_seed_all(seed)

    # Load model
    model = GPT4.from_pretrained(model_dir)
    if num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpus)))
    model.eval()

    # Load data
    data_loader = GPT4DataLoader(data_dir, batch_size, max_seq_len)

    # Evaluate
    total_loss = 0.0
    for _ in range(num_eval_batches):
        batch = data_loader.next_batch()
        loss = model(batch, num_samples=num_samples).mean()
        total_loss += loss.item()
        if verbose:
            logger.info('Batch loss: %.4f', loss.item())

    avg_loss = total_loss / num_eval_batches
    avg_perplexity = np.exp(avg_loss)
    logger.info('Average perplexity: %.4f', avg_perplexity)

    return avg_perplexity