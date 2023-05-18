import os
import json
import logging
import numpy as np
import torch

from typing import List, Optional, Union

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """
    Sets the random seed for all random number generators used in OpenAI GPT-4.

    Args:
        seed (int): The random seed to use.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_json(file_path: str) -> dict:
    """
    Loads a JSON file from the given file path.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The loaded JSON file.
    """
    with open(file_path, "r") as f:
        return json.load(f)

def save_json(data: dict, file_path: str):
    """
    Saves a JSON file to the given file path.

    Args:
        data (dict): The data to save.
        file_path (str): The path to the JSON file.
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def get_files_in_dir(dir_path: str, extension: Optional[str] = None) -> List[str]:
    """
    Gets a list of files in the given directory path.

    Args:
        dir_path (str): The path to the directory.
        extension (str, optional): The file extension to filter by.

    Returns:
        List[str]: The list of files in the directory.
    """
    files = os.listdir(dir_path)
    if extension is not None:
        files = [f for f in files if f.endswith(extension)]
    return files

def get_device(cuda_id: Optional[Union[int, str]] = None) -> torch.device:
    """
    Gets the device to use for training.

    Args:
        cuda_id (int, str, optional): The CUDA device ID to use. If None, the CPU is used.

    Returns:
        torch.device: The device to use.
    """
    if cuda_id is None:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(cuda_id))
    return device
