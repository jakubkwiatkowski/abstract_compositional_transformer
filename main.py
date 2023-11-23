import pyarrow
pyarrow.PyExtensionType.set_auto_load(True)

import pyarrow_hotfix
pyarrow_hotfix.uninstall()

from dataclasses import asdict, dataclass
import tensorflow as tf
from core_tools.start import keras_get_item

keras_get_item()
tf.experimental.numpy.experimental_enable_numpy_behavior(prefer_float32=True)

import numpy as np
import fire
from datasets import load_dataset

from core_tools.tmp import tmp_compile
from core_tools.core import  as_dict

from raven_utils.models.transformer import get_rav_trans, rav_select_model
from raven_utils.params import DirectChoiceMakerParameters, TaskTokenizerParameters, LearnableChoiceMakerParameters, \
    Latent3ContrastiveLearnableChoiceMakerParameters, RowTokenizerParameters, PanelTokenizerParameters

DATASET_REPOSITORY = "jkwiatkowski/raven"


def get_tf_dataset(
        data_split: str = "val",
        batch_size: int = 64
):
    data = load_dataset(DATASET_REPOSITORY, split=data_split)

    index = 0
    sample_data = data[index:index + 1]['inputs']
    sample_data = {
        'inputs': np.asarray(sample_data, dtype="uint8"),
        'index': np.zeros(shape=(1, 1), dtype="uint8") + 8,
        'target': np.zeros(shape=(1, 16, 113), dtype="int8"),
    }

    tf_data = data.to_tf_dataset(batch_size=batch_size)

    def change_dtype(element):
        element = {
            'inputs': tf.cast(element['inputs'], dtype="uint8"),
            'index': tf.cast(element['index'], dtype="uint8")[:, None],
            'target': tf.cast(element['target'], dtype="int8"),
        }
        return element

    tf_data = tf_data.map(change_dtype)
    return tf_data, sample_data


def property_prediction(
        phase: str = "train",
        tokenizer: str = "task",
        masking: str = "query",
        data_split: str = "train",
        load_weights: str = None,
        save_weights: str = "model/weights",
        batch_size: int = 64,
        epochs: int = 200,
        **kwargs,

):
    """Trains or evaluates a property prediction model for RAVEN matrices.

    Args:
        phase (str, optional): The phase of the model [train, eval].
        tokenizer (str, optional): The tokenizer to use [task, row, panel].
        masking (str, optional): The masking strategy [query, random].
        data_split (str, optional): The data split to use [train, val, test].
        load_weights (str, optional): The path to load weights from.
        save_weights (str, optional): The path to save weights.
        batch_size (int, optional): The batch size.
        epochs (int, optional): The number of epochs.

    """
    if isinstance(tokenizer, str):
        if tokenizer == "task":
            tokenizer = TaskTokenizerParameters()
        elif tokenizer == "row":
            tokenizer = RowTokenizerParameters()
        elif tokenizer == "panel":
            tokenizer = PanelTokenizerParameters()

    tf_data, sample_data = get_tf_dataset(
        data_split=data_split,
        batch_size=batch_size
    )
    if masking == "query":
        masking = "last"
    tokenizer.mask = masking
    model = get_rav_trans(
        sample_data,
        **{
            **asdict(tokenizer),
            **kwargs
        }
    )

    run_model(epochs, load_weights, save_weights, model, phase, sample_data, tf_data)


def choice_maker_(
        phase: str = "train",
        tokenizer: str = "task",
        choice_maker: str = "ccm",
        data_split: str = "train",
        act_load_weights: str = None,
        load_weights: str = None,
        save_weights: str = "model/weights",
        batch_size: int = 64,
        epochs: int = 200,
        **kwargs
):
    """Trains or evaluates a choice maker model for RAVEN matrices.

    Args:
        phase (str, optional): The phase of the model [train, eval].
        tokenizer (str, optional): The tokenizer to use [task, row, panel].
        choice_maker (str, optional): The type of choice making [dcm, lcm, ccm].
        data_split (str, optional): The data split to use [train, val, test].
        act_load_weights (str, optional): The path to load property prediction weights from.
        load_weights (str, optional): The path to load weights from.
        save_weights (str, optional): The path to save weights.
        batch_size (int, optional): The batch size.
        epochs (int, optional): The number of epochs.

    """
    if isinstance(tokenizer, str):
        if tokenizer == "task":
            tokenizer = TaskTokenizerParameters()
        elif tokenizer == "row":
            tokenizer = RowTokenizerParameters()
        elif tokenizer == "panel":
            tokenizer = PanelTokenizerParameters()

    if isinstance(choice_maker, str):
        if choice_maker == "dcm":
            choice_maker = DirectChoiceMakerParameters
        elif choice_maker == "lcm":
            choice_maker = LearnableChoiceMakerParameters
        elif choice_maker == "ccm":
            choice_maker = Latent3ContrastiveLearnableChoiceMakerParameters

    tf_data, sample_data = get_tf_dataset(
        data_split=data_split,
        batch_size=batch_size
    )

    @dataclass
    class TmpRaven(tokenizer.__class__, choice_maker):
        pass

    p = TmpRaven()

    if choice_maker == LearnableChoiceMakerParameters:
        p.additional_copy = True
        p.mask = "last"
        # p.additional_copy = True
        p.predictor = [p.predictor_size] * p.predictor_no
        if p.predictor_norm is not None:
            p.predictor_pre = p.predictor_norm[0]
            p.predictor_post = p.predictor_norm[1]

    model = rav_select_model(
        sample_data,
        load_weights=act_load_weights,
        **{
            **as_dict(p),
            **kwargs
        }
    )

    if choice_maker == LearnableChoiceMakerParameters:
        if p.train_only_predictor:
            model[0, 0].trainable = False
            model[0, 1].trainable = False
            if p.train_only_predictor > 1:
                model[0, 1, -2].trainable = True

    run_model(epochs, load_weights, save_weights, model, phase, sample_data, tf_data)


def run_model(epochs, load_weights, save_weights, model, phase, sample_data, tf_data):
    model(sample_data)
    if load_weights:
        model.load_weights(load_weights)
    tmp_compile(model)
    if phase == "train":
        result = model.fit(tf_data, epochs=epochs)
        if save_weights:
            model.save_weights(save_weights)
        result = result.history
    else:
        result = model.evaluate(tf_data, return_dict=True)
    print(result)


if __name__ == '__main__':
    fire.Fire({
        "pp": property_prediction,
        "cm": choice_maker_,

    })
