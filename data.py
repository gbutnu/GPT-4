import os
import json

import numpy as np

from tokenization import Tokenizer

class Data:
    """
    Class for loading and preprocessing data for OpenAI GPT-4.
    """

    def __init__(self, data_dir, tokenizer, max_seq_length):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.train_data = None
        self.valid_data = None
        self.test_data = None

    def load_data(self):
        """
        Loads the data from the data directory.
        """
        # Load the training data
        train_data_path = os.path.join(self.data_dir, 'train.json')
        with open(train_data_path, 'r') as f:
            self.train_data = json.load(f)

        # Load the validation data
        valid_data_path = os.path.join(self.data_dir, 'valid.json')
        with open(valid_data_path, 'r') as f:
            self.valid_data = json.load(f)

        # Load the test data
        test_data_path = os.path.join(self.data_dir, 'test.json')
        with open(test_data_path, 'r') as f:
            self.test_data = json.load(f)

    def preprocess_data(self):
        """
        Preprocesses the data by tokenizing it and padding it to the maximum sequence length.
        """
        # Tokenize the training data
        self.train_data['input_ids'] = self.tokenizer.batch_encode_plus(
            self.train_data['text'],
            max_length=self.max_seq_length,
            pad_to_max_length=True
        )['input_ids']

        # Tokenize the validation data
        self.valid_data['input_ids'] = self.tokenizer.batch_encode_plus(
            self.valid_data['text'],
            max_length=self.max_seq_length,
            pad_to_max_length=True
        )['input_ids']

        # Tokenize the test data
        self.test_data['input_ids'] = self.tokenizer.batch_encode_plus(
            self.test_data['text'],
            max_length=self.max_seq_length,
            pad_to_max_length=True
        )['input_ids']

    def get_train_data(self):
        """
        Returns the training data.
        """
        return self.train_data

    def get_valid_data(self):
        """
        Returns the validation data.
        """
        return self.valid_data

    def get_test_data(self):
        """
        Returns the test data.
        """
        return self.test_data