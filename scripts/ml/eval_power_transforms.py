import functools
import os
from pathlib import Path

import keras
import keras.ops as k_ops
import matplotlib.pyplot as plt
from resolv_mir import NoteSequence
from resolv_ml.utilities.statistic.power_transforms import BoxCox, YeoJohnson
from resolv_pipelines.data.loaders import TFRecordLoader
from resolv_pipelines.data.representation.mir import PitchSequenceRepresentation

from scripts.utilities.constants import Paths

DATASET_FOLDER = "4bars_melodies_distinct/lakh_midi-v1.0.0-clean/midi"
DATASET_FILE_PATTERN = 'pitchseq/pitchseq-*.tfrecord'
DATASET_PATH = str(Paths.REPRESENTATION_DATASETS_DIR / DATASET_FOLDER / DATASET_FILE_PATTERN)
BATCH_SIZE = 64
BOX_COX_LAMBDA = 0.33
YEO_JOHNSON_LAMBDA = -2.22
HIST_OUTPUT_PATH = Paths.REPRESENTATION_DATASETS_DIR / DATASET_FOLDER / 'power_transforms'
HISTOGRAM_BINS = [20, 30, 40, 50, 60, 70, 80]


def load_dataset(attribute: str):
    representation = PitchSequenceRepresentation(sequence_length=64)
    tfrecord_loader = TFRecordLoader(
        file_pattern=DATASET_PATH,
        parse_fn=functools.partial(
            representation.parse_example,
            attributes_to_parse=[attribute],
            parse_sequence_feature=False
        ),
        map_fn=lambda x, y: x[attribute],
        batch_size=BATCH_SIZE,
        batch_drop_reminder=True,
        deterministic=True
    )
    return tfrecord_loader.load_dataset()


def plot_distribution(x, output_path: Path, power_transform_id: str, lmbda: int, attribute: str):
    output_path = output_path / power_transform_id
    output_path.mkdir(parents=True, exist_ok=True)
    pt_title = power_transform_id.replace("_", " ").capitalize()
    attr_title = attribute.replace("_", " ").capitalize()
    for bins in HISTOGRAM_BINS:
        plt.hist(x, bins=bins, color='blue', alpha=0.7)
        plt.suptitle(f'{pt_title} - {attr_title}')
        plt.title(f'lambda = {lmbda} - {bins} bins')
        plt.savefig(f'{str(output_path)}/{attribute}_histogram_{lmbda}_{bins}.png', format='png', dpi=300)
        plt.close()


def test_box_cox_transform(attribute: str):
    class BoxCoxModel(keras.Model):

        def __init__(self, lmbda: int):
            super(BoxCoxModel, self).__init__()
            self._lmbda = lmbda

        # noinspection PyAttributeOutsideInit
        def build(self, input_shape):
            self._box_cox_layer = BoxCox(lambda_init=self._lmbda)

        def call(self, inputs):
            pt_out = self._box_cox_layer(inputs)
            pt_out_std = k_ops.divide(k_ops.subtract(pt_out, k_ops.mean(pt_out)), k_ops.std(pt_out))
            return pt_out_std

    dataset = load_dataset(attribute)
    box_cox_model = BoxCoxModel(lmbda=BOX_COX_LAMBDA)
    box_cox_model.trainable = False
    box_cox_model.compile(run_eagerly=True)  # TODO - fix pt exception handling to run with autograph
    power_transform_attribute = box_cox_model.predict(dataset)
    plot_distribution(power_transform_attribute, HIST_OUTPUT_PATH, "box-cox", BOX_COX_LAMBDA, attribute_name)


def test_yeo_johnson_transform(attribute: str):
    class YeoJohnsonModel(keras.Model):

        def __init__(self, lmbda: int):
            super(YeoJohnsonModel, self).__init__()
            self._lmbda = lmbda

        # noinspection PyAttributeOutsideInit
        def build(self, input_shape):
            self._yeo_johnson_layer = YeoJohnson(lambda_init=self._lmbda)

        def call(self, inputs):
            pt_out = self._yeo_johnson_layer(inputs)
            pt_out_std = k_ops.divide(k_ops.subtract(pt_out, k_ops.mean(pt_out)), k_ops.std(pt_out))
            return pt_out_std

    dataset = load_dataset(attribute)
    yeo_johnson_model = YeoJohnsonModel(lmbda=YEO_JOHNSON_LAMBDA)
    yeo_johnson_model.trainable = False
    yeo_johnson_model.compile(run_eagerly=True)  # TODO - fix pt exception handling to run with autograph
    power_transform_attribute = yeo_johnson_model.predict(dataset)
    plot_distribution(power_transform_attribute, HIST_OUTPUT_PATH, "yeo-johnson", YEO_JOHNSON_LAMBDA, attribute_name)


if __name__ == '__main__':
    os.environ["KERAS_BACKEND"] = "tensorflow"
    HIST_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    for field in NoteSequence.SequenceAttributes.DESCRIPTOR.fields:
        attribute_name = field.name
        test_box_cox_transform(attribute_name)
        test_yeo_johnson_transform(attribute_name)
