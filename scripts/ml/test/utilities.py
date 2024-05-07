import functools

import keras
import tensorflow as tf
from resolv_pipelines.data.loaders import TFRecordLoader
from resolv_pipelines.data.representation.mir import PitchSequenceRepresentation


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
