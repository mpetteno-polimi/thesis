import argparse
import functools

import keras
import tensorflow as tf
from resolv_mir.note_sequence.attributes import compute_attribute
from resolv_mir.note_sequence.representations.sequence import HOLD_NOTE_SYMBOL
from resolv_pipelines.data.loaders import TFRecordLoader
from resolv_pipelines.data.representation.mir import PitchSequenceRepresentation


def compute_sequences_attributes(decoded_sequences, attribute_name: str, sequence_length: int):
    decoded_ns_attributes = []
    representation = PitchSequenceRepresentation(sequence_length)
    hold_note_start_seq_count = 0
    for decoded_sequence in decoded_sequences:
        if decoded_sequence[0] == HOLD_NOTE_SYMBOL:
            hold_note_start_seq_count += 1
        decoded_note_sequence = representation.to_canonical_format(decoded_sequence, attributes=None)
        decoded_ns_attribute = compute_attribute(decoded_note_sequence, attribute_name)
        decoded_ns_attributes.append(decoded_ns_attribute)
    return decoded_ns_attributes, hold_note_start_seq_count


def load_dataset(dataset_path: str,
                 sequence_length: int,
                 attribute: str,
                 batch_size: int,
                 parse_sequence_feature: bool = True):

    def map_fn(ctx, seq):
        # Since there may be 0 valued attributes, add an epsilon to everything in order to avoid problems with the
        # BoxCox Transform computation
        attributes = ctx[attribute] + keras.backend.epsilon()
        if parse_sequence_feature:
            input_seq = tf.transpose(seq["pitch_seq"])
            target = input_seq
            return (input_seq, attributes), target
        else:
            return attributes

    representation = PitchSequenceRepresentation(sequence_length=sequence_length)
    tfrecord_loader = TFRecordLoader(
        file_pattern=dataset_path,
        parse_fn=functools.partial(
            representation.parse_example,
            attributes_to_parse=[attribute],
            parse_sequence_feature=parse_sequence_feature
        ),
        map_fn=map_fn,
        batch_size=batch_size,
        batch_drop_reminder=True
    )
    return tfrecord_loader.load_dataset()


def get_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model-path', required=True, help='Path to the dataset containing SequenceExample with the '
                                                            'attributes to evaluate saved in the context.')
    parser.add_argument('--test-dataset-path', required=True,
                        help='Path to the dataset containing the test SequenceExample.')
    parser.add_argument('--dataset-cardinality', help='Cardinality of the test dataset.', required=True, type=int)
    parser.add_argument('--sequence-length', help='Length of the sequences in the dataset.', required=True, type=int)
    parser.add_argument('--attributes', nargs='+', help='Attributes to evaluate.', required=True)
    parser.add_argument('--regularized-dimension', help='Index of the latent code regularized dimension.',
                        required=False, type=int, default=0)
    parser.add_argument('--output-path', help='Path where the histograms and generated MIDI files will be saved.',
                        required=True)
    parser.add_argument('--batch-size', help='Batch size.', required=False, default=64, type=int,
                        choices=[32, 64, 128, 256, 512])
    parser.add_argument('--seed', help='Seed for random initializers.', required=False, type=int)
    parser.add_argument('--logging-level', help='Set the logging level.', default="INFO", required=False,
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO"])
    return parser
