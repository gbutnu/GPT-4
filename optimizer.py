import os
import json
import logging
import numpy as np
import tensorflow as tf

from . import model_utils

logger = logging.getLogger(__name__)


class ModelOptimizer(object):
    """Model optimizer for GPT-4."""

    def __init__(self, model_dir, config):
        """Initializes the model optimizer.

        Args:
          model_dir: The directory where the model is stored.
          config: The model configuration.
        """
        self.model_dir = model_dir
        self.config = config

    def optimize(self, global_step):
        """Optimizes the model.

        Args:
          global_step: The global step of the model.
        """
        # Load the model weights.
        weights_path = os.path.join(self.model_dir, 'model.ckpt')
        weights = model_utils.load_weights(weights_path)

        # Create the optimizer.
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.config.learning_rate)

        # Compute the gradients.
        grads_and_vars = optimizer.compute_gradients(
            self.config.loss, var_list=weights.values())

        # Apply the gradients.
        train_op = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)

        # Save the weights.
        model_utils.save_weights(weights_path, weights)

        return train_op