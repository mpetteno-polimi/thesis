import functools
import os
from pathlib import Path
from typing import Type, List

import keras
import keras.ops as k_ops
import matplotlib.pyplot as plt
import numpy as np
from resolv_mir import NoteSequence
from resolv_ml.utilities.statistic.power_transforms import BoxCox, YeoJohnson, PowerTransform
from resolv_pipelines.data.loaders import TFRecordLoader
from resolv_pipelines.data.representation.mir import PitchSequenceRepresentation

from scripts.utilities.constants import Paths

# DATASET CONFIG
DATASET_FOLDER = "4bars_melodies_distinct/lakh_midi-v1.0.0-clean/midi"
DATASET_FILE_PATTERN = 'pitchseq/pitchseq-*.tfrecord'
DATASET_PATH = str(Paths.REPRESENTATION_DATASETS_DIR / DATASET_FOLDER / DATASET_FILE_PATTERN)
BATCH_SIZE = 64
# HISTOGRAM CONFIG
HIST_OUTPUT_PATH = Paths.REPRESENTATION_DATASETS_DIR / DATASET_FOLDER / 'power_transforms'
HISTOGRAM_BINS = [60]
# POWER TRANSFORM CONFIG
LAMBDA_MIN = -3
LAMBDA_MAX = 3
LAMBDA_STEP = 0.25


def eval_power_transforms(power_transform_classes: List[Type[PowerTransform]]) -> List[float]:
    class PowerTransformModel(keras.Model):

        def __init__(self, pt_class: Type[PowerTransform], lambd: int):
            super(PowerTransformModel, self).__init__()
            self._power_transform_class = pt_class
            self._lmbda = lambd

        # noinspection PyAttributeOutsideInit
        def build(self, input_shape):
            self._pt_layer = self._power_transform_class(lambda_init=self._lmbda)
            self.name = self._pt_layer.name

        def call(self, inputs):
            pt_out = self._pt_layer(inputs)
            pt_out_mean = k_ops.mean(pt_out)
            pt_out_std = k_ops.std(pt_out)
            pt_out_norm = (pt_out - pt_out_mean) / pt_out_std
            kl_div_batch = -0.5 * k_ops.sum((1 + k_ops.log(pt_out_std) - k_ops.square(pt_out_mean) - pt_out_std))
            return pt_out_norm, kl_div_batch

    lambds = k_ops.arange(LAMBDA_MIN, LAMBDA_MAX, LAMBDA_STEP)
    for field in NoteSequence.SequenceAttributes.DESCRIPTOR.fields:
        attribute = field.name
        for power_transform_class in power_transform_classes:
            power_transform_id = power_transform_class.__name__
            output_path = HIST_OUTPUT_PATH / power_transform_id.lower() / attribute
            output_path.mkdir(parents=True, exist_ok=True)
            kl_divs = []
            for lmbda in lambds:
                dataset = _load_dataset(attribute)
                pt_model = PowerTransformModel(power_transform_class, lmbda)
                pt_model.trainable = False
                pt_model.compile()
                output, kl_div_batches = pt_model.predict(dataset)
                kld = k_ops.mean(kl_div_batches)
                _plot_distribution(output, output_path, power_transform_id, lmbda, kld, attribute)
                kl_divs.append(kld)
            # Save KLD and lambda values and plot KLD as a function of lambda
            np.save(output_path / 'numpy_klds.npy', kl_divs)
            _plot_kld_lambda_fn(lambds, kl_divs, output_path, power_transform_id, attribute)


def _load_dataset(attribute: str):
    representation = PitchSequenceRepresentation(sequence_length=64)
    tfrecord_loader = TFRecordLoader(
        file_pattern=DATASET_PATH,
        parse_fn=functools.partial(
            representation.parse_example,
            attributes_to_parse=[attribute],
            parse_sequence_feature=False
        ),
        # Since there may be 0 valued attributes, add an epsilon to everything in order to avoid problems with the
        # BoxCox Transform computation
        map_fn=lambda x, _: x[attribute] + keras.backend.epsilon(),
        batch_size=BATCH_SIZE,
        batch_drop_reminder=True,
        deterministic=True
    )
    return tfrecord_loader.load_dataset()


def _plot_distribution(x, output_path: Path, power_transform_id: str, lmbda: float, kl_div: float, attribute: str):
    pt_title = power_transform_id
    attr_title = attribute.replace("_", " ").capitalize()
    for bins in HISTOGRAM_BINS:
        filename = f'{str(output_path)}/histogram_{lmbda:.2f}_{kl_div:.2f}_{bins}.png'
        plt.hist(x, bins=bins, color='blue', alpha=0.7)
        plt.suptitle(f'{pt_title} - {attr_title}')
        plt.title(r'$\lambda$ = ' + f'{lmbda:.2f} - KLD = {kl_div:.2f} - Bins = {bins}')
        plt.grid(linestyle=':')
        plt.savefig(filename, format='png', dpi=300)
        plt.close()


def _plot_kld_lambda_fn(lambds, klds, output_path: Path, power_transform_id: str, attribute: str):
    pt_title = power_transform_id
    attr_title = attribute.replace("_", " ").capitalize()
    filename = f'{str(output_path)}/kld_vs_lmbda_plot.png'
    plt.plot(lambds, klds)
    plt.title(f'{pt_title} - {attr_title}')
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'KLD($\lambda$)')
    plt.grid(linestyle=':')
    plt.savefig(filename, format='png', dpi=300)
    plt.close()


if __name__ == '__main__':
    os.environ["KERAS_BACKEND"] = "tensorflow"
    eval_power_transforms([BoxCox, YeoJohnson])
