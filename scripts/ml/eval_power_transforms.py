import functools
import os
from pathlib import Path

import keras
import matplotlib.pyplot as plt
from resolv_mir import NoteSequence
from resolv_ml.utilities.statistic.power_transforms import BoxCox, YeoJohnson
from resolv_pipelines.data.loaders import TFRecordLoader
from resolv_pipelines.data.representation.mir import PitchSequenceRepresentation

from scripts.utilities.constants import Paths

DATASET_FOLDER = "4bars_melodies_distinct/lakh_midi-v1.0.0-clean/midi"
DATASET_FILE_PATTERN = 'representation-*.tfrecord'
DATASET_PATH = str(Paths.REPRESENTATION_DATASETS_DIR / DATASET_FOLDER / DATASET_FILE_PATTERN)
BATCH_SIZE = 64
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
        batch_size=BATCH_SIZE,
        batch_drop_reminder=True,
        deterministic=True
    )
    return tfrecord_loader.load_dataset()


def plot_distribution(x, output_path: Path, attribute: str):
    output_path.mkdir(parents=True, exist_ok=True)
    for bins in HISTOGRAM_BINS:
        plt.hist(x, bins=bins, color='blue', alpha=0.7)
        plt.title(f'{attribute.replace("_", " ")} - {bins} bins')
        plt.savefig(f'{str(output_path)}/{attribute}_histogram_{bins}.png', format='png', dpi=300)
        plt.close()


def test_box_cox_transform(attribute: str):
    class BoxCoxModel(keras.Model):

        # noinspection PyAttributeOutsideInit
        def build(self, input_shape):
            self._box_cox_layer = BoxCox(lambda_init=0.33, batch_norm=keras.layers.BatchNormalization())

        def call(self, inputs):
            return self._box_cox_layer(inputs)

    dataset = load_dataset(attribute)
    box_cox_model = BoxCoxModel()
    box_cox_model.trainable = False
    box_cox_model.compile(run_eagerly=True)  # TODO - fix pt exception handling to run with autograph
    power_transform_attribute = box_cox_model.predict(dataset)
    hist_output_path = HIST_OUTPUT_PATH / "box-cox"
    plot_distribution(power_transform_attribute, hist_output_path, attribute_name)


def test_yeo_johnson_transform(attribute: str):
    class YeoJohnsonModel(keras.Model):

        # noinspection PyAttributeOutsideInit
        def build(self, input_shape):
            self._yeo_johnson_layer = YeoJohnson(lambda_init=-2.22, batch_norm=keras.layers.BatchNormalization())

        def call(self, inputs):
            return self._yeo_johnson_layer(inputs)

    dataset = load_dataset(attribute)
    yeo_johnson_model = YeoJohnsonModel()
    yeo_johnson_model.trainable = False
    yeo_johnson_model.compile(run_eagerly=True)  # TODO - fix pt exception handling to run with autograph
    power_transform_attribute = yeo_johnson_model.predict(dataset)
    hist_output_path = HIST_OUTPUT_PATH / "yeo-johnson"
    plot_distribution(power_transform_attribute, hist_output_path, attribute_name)


if __name__ == '__main__':
    os.environ["KERAS_BACKEND"] = "tensorflow"
    HIST_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    for field in NoteSequence.SequenceAttributes.DESCRIPTOR.fields:
        attribute_name = field.name
        test_box_cox_transform(attribute_name)
        test_yeo_johnson_transform(attribute_name)
