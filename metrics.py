import os
import json
import numpy as np
import tensorflow as tf

from . import metrics

def compute_metrics(model, dataset, batch_size, num_batches,
                    metrics_dir, metrics_prefix,
                    metrics_to_compute=None,
                    metrics_config=None):
    """Computes metrics for a given model and dataset.

    Args:
        model: A tf.keras.Model instance.
        dataset: A tf.data.Dataset instance.
        batch_size: The batch size to use when computing metrics.
        num_batches: The number of batches to use when computing metrics.
        metrics_dir: The directory to save the metrics to.
        metrics_prefix: The prefix to use when saving the metrics.
        metrics_to_compute: A list of metrics to compute. If None, all
            available metrics will be computed.
        metrics_config: A dict of configuration parameters for the metrics.
            If None, the default configuration will be used.

    Returns:
        A dict of computed metrics.
    """
    if metrics_to_compute is None:
        metrics_to_compute = metrics.get_available_metrics()

    if metrics_config is None:
        metrics_config = {}

    metrics_dict = {}
    for metric_name in metrics_to_compute:
        metric_fn = metrics.get_metric_fn(metric_name)
        metric_config = metrics_config.get(metric_name, {})
        metric_value = metric_fn(model, dataset, batch_size, num_batches,
                                 metric_config)
        metrics_dict[metric_name] = metric_value

    if metrics_dir is not None:
        metrics_path = os.path.join(metrics_dir,
                                    metrics_prefix + '_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)

    return metrics_dict