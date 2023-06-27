import tensorflow as tf
tf.experimental.numpy.experimental_enable_numpy_behavior(prefer_float32=True)

from huggingface_hub import from_pretrained_keras
from datasets import load_dataset

repo = "jkwiatkowski/raven"

data = load_dataset(repo, split="val")
properties = load_dataset(repo + "_properties", split="val")
model = from_pretrained_keras(repo)

START_IMAGE = 12000
