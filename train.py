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

model = get_rav_trans(
    d,
    **asdict(TaskTokenizerParameters())
)


def generator():
    for d in data:
        rd = {
            'inputs': np.asarray(d['inputs'], dtype="uint8"),
            'index': np.asarray(d['index'], dtype="uint8")[..., None],
            'target': np.asarray(d['target'], dtype="uint8"),
        }
        yield rd


tf_data = tf.data.Dataset.from_generator(generator,
                                         output_shapes=
                                         {'inputs': (None, 16, 160, 160), 'index': (None, 1), 'target': (None, 16, 113)}
                                         ,
                                         output_types={
                                             'inputs': "uint8",
                                             'index': "uint8",
                                             'target': "uint8",
                                         },
                                         )
model.compile()
model.fit(tf_data)
model(data[0])

print("End")
