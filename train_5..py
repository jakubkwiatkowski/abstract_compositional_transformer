from core_tools.start import keras_get_item

keras_get_item()

from dataclasses import asdict, dataclass

import tensorflow as tf

tf.experimental.numpy.experimental_enable_numpy_behavior(prefer_float32=True)

from huggingface_hub import from_pretrained_keras
from datasets import load_dataset
import numpy as np
from core_tools.tmp import tmp_compile

from raven_utils.models.transformer import get_rav_trans, rav_select_model
from raven_utils.params import DirectChoiceMakerParameters, TaskTokenizerParameters, LearnableChoiceMakerParameters, Latent3ContrastiveLearnableChoiceMakerParameters

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


@dataclass
class TmpRaven(Latent3ContrastiveLearnableChoiceMakerParameters, TaskTokenizerParameters):
    pass


p = TmpRaven()

if hasattr(p, "predictor_size") and hasattr(p, "predictor_no"):
    p.predictor = [p.predictor_size] * p.predictor_no

if hasattr(p, "predictor_norm") and p.predictor_norm is not None:
    p.predictor_pre = p.predictor_norm[0]
    p.predictor_post = p.predictor_norm[1]

model = rav_select_model(
    d,
    load_weights=None,
    **asdict(p)
)

model(d)
# model.load_weights("/home/jkwiatkowski/all/best/model/51f18f4848c1450188bed731f98552b6/weights_13-0.79")
tmp_compile(model)
result = model.evaluate(tf_data, return_dict=True)
# model.fit(tf_data)
# model(data[0])

print(result)
