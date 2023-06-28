from dataclasses import asdict

import tensorflow as tf

tf.experimental.numpy.experimental_enable_numpy_behavior(prefer_float32=True)

from huggingface_hub import from_pretrained_keras
from datasets import load_dataset
import numpy as np

from raven_utils.models.transformer import get_rav_trans
from raven_utils.params import TaskTokenizerParameters

repo = "jkwiatkowski/raven"

data = load_dataset(repo, split="val")
properties = load_dataset(repo + "_properties", split="val")

index = 0
d = data[index:index + 1]['inputs']
d = {
    'inputs': np.asarray(d, dtype="uint8"),
    'index': np.zeros(shape=(1, 1), dtype="uint8") + 8,
    'target': np.zeros(shape=(1, 16, 113), dtype="int8"),
}

tf_data = data.to_tf_dataset(batch_size=64)


def change_dtype(element):
    element = {
        'inputs': tf.cast(element['inputs'], dtype="uint8"),
        'index': tf.cast(element['index'], dtype="uint8")[:, None],
        'target': tf.cast(element['target'], dtype="int8"),
    }
    return element


tf_data = tf_data.map(change_dtype)

for x in tf_data.take(5):
    print(x)

model = get_rav_trans(
    d,
    **asdict(TaskTokenizerParameters())
)
model.compile()
model.fit(tf_data)
# model(data[0])

print("End")
