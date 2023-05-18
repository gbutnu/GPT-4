import os
import shutil
import time

import numpy as np
import tensorflow as tf

from gpt_4.src.model import model_utils


def get_checkpoint_dir(model_dir):
    """Returns the directory where checkpoints are stored."""
    return os.path.join(model_dir, "checkpoints")


def get_checkpoint_path(model_dir, step):
    """Returns the path to the checkpoint file for the given step."""
    return os.path.join(get_checkpoint_dir(model_dir), "model.ckpt-{}".format(step))


def get_checkpoint_state(model_dir):
    """Returns the checkpoint state for the given model directory."""
    ckpt = tf.train.get_checkpoint_state(get_checkpoint_dir(model_dir))
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        return ckpt_name.split("-")[-1]
    return None


def save_checkpoint(model_dir, step, saver, sess):
    """Saves a checkpoint for the given step."""
    checkpoint_path = get_checkpoint_path(model_dir, step)
    saver.save(sess, checkpoint_path)


def restore_checkpoint(model_dir, saver, sess):
    """Restores the latest checkpoint from the given model directory."""
    ckpt_state = get_checkpoint_state(model_dir)
    if ckpt_state is None:
        return False
    saver.restore(sess, get_checkpoint_path(model_dir, ckpt_state))
    return True


def delete_checkpoint(model_dir):
    """Deletes the checkpoint for the given model directory."""
    ckpt_state = get_checkpoint_state(model_dir)
    if ckpt_state is None:
        return
    os.remove(get_checkpoint_path(model_dir, ckpt_state))


def save_checkpoint_with_metadata(model_dir, step, saver, sess):
    """Saves a checkpoint with metadata for the given step."""
    checkpoint_path = get_checkpoint_path(model_dir, step)
    saver.save(sess, checkpoint_path)

    # Save the step and timestamp in a separate file.
    metadata_path = os.path.join(get_checkpoint_dir(model_dir), "metadata.txt")
    with open(metadata_path, "w") as f:
        f.write("step={}\ntimestamp={}".format(step, int(time.time())))


def restore_checkpoint_with_metadata(model_dir, saver, sess):
    """Restores the latest checkpoint with metadata from the given model directory."""
    ckpt_state = get_checkpoint_state(model_dir)
    if ckpt_state is None:
        return False
    saver.restore(sess, get_checkpoint_path(model_dir, ckpt_state))

    # Load the step and timestamp from the metadata file.
    metadata_path = os.path.join(get_checkpoint_dir(model_dir), "metadata.txt")
    with open(metadata_path, "r") as f:
        for line in f:
            if line.startswith("step="):
                step = int(line.split("=")[1])
            elif line.startswith("timestamp="):
                timestamp = int(line.split("=")[1])
    return step, timestamp


def delete_checkpoint_with_metadata(model_dir):
    """Deletes the checkpoint and metadata for the given model directory."""
    ckpt_state = get_checkpoint_state(model_dir)
    if ckpt_state is None:
        return
    os.remove(get_checkpoint_path(model_dir, ckpt_state))
    os.remove(os.path.join(get_checkpoint_dir(model_dir), "metadata.txt"))


def copy_checkpoint(model_dir, dest_dir):
    """Copies the checkpoint from the given model directory to the destination directory."""
    ckpt_state = get_checkpoint_state(model_dir)
    if ckpt_state is None:
        return
    src_path = get_checkpoint_path(model_dir, ckpt_state)
    dest_path = get_checkpoint_path(dest_dir, ckpt_state)
    shutil.copyfile(src_path, dest_path)


def copy_checkpoint_with_metadata(model_dir, dest_dir):
    """Copies the checkpoint and metadata from the given model directory to the destination directory."""
    ckpt_state = get_checkpoint_state(model_dir)
    if ckpt_state is None:
        return
    src_path = get_checkpoint_path(model_dir, ckpt_state)
    dest_path = get_checkpoint_path(dest_dir, ckpt_state)
    shutil.copyfile(src_path, dest_path)

    src_metadata_path = os.path.join(get_checkpoint_dir(model_dir), "metadata.txt")
    dest_metadata_path = os.path.join(get_checkpoint_dir(dest_dir), "metadata.txt")
    shutil.copyfile(src_metadata_path, dest_metadata_path)


def get_checkpoint_variables(model_dir):
    """Returns the variables in the checkpoint for the given model directory."""
    ckpt_state = get_checkpoint_state(model_dir)
    if ckpt_state is None:
        return None
    reader = tf.train.NewCheckpointReader(get_checkpoint_path(model_dir, ckpt_state))
    return reader.get_variable_to_shape_map()


def get_checkpoint_variable_values(model_dir):
    """Returns the values of the variables in the checkpoint for the given model directory."""
    ckpt_state = get_checkpoint_state(model_dir)
    if ckpt_state is None:
        return None
    reader = tf.train.NewCheckpointReader(get_checkpoint_path(model_dir, ckpt_state))
    return reader.get_variable_to_dtype_map()


def get_checkpoint_variable_values_as_numpy_arrays(model_dir):
    """Returns the values of the variables in the checkpoint as numpy arrays for the given model directory."""
    ckpt_state = get


"""# Copyright 2020 OpenAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import logging
import torch

logger = logging.getLogger(__name__)


class ModelCheckpoint:
    """
    Model checkpointing utility.
    """

    def __init__(self, model, save_dir, save_freq=1):
        self.model = model
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.step = 0

    def save(self, step=None):
        """
        Save the model.
        """
        if step is None:
            step = self.step
        else:
            self.step = step

        if self.save_freq > 0 and step % self.save_freq == 0:
            save_path = os.path.join(self.save_dir, "model_step_{}.pt".format(step))
            logger.info("Saving model checkpoint to {}".format(save_path))
            torch.save(self.model.state_dict(), save_path)

    def load(self, step=None):
        """
        Load the model.
        """
        if step is None:
            step = self.step
        else:
            self.step = step

        save_path = os.path.join(self.save_dir, "model_step_{}.pt".format(step))
        logger.info("Loading model checkpoint from {}".format(save_path))
        self.model.load_state_dict(torch.load(save_path))
"

