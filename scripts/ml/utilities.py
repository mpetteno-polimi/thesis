import functools
import json

import keras
import tensorflow as tf
from resolv_ml.models.dlvm.vae.ar_vae import AttributeRegularizedVAE, AttributeRegularizationLayer
from resolv_ml.models.seq2seq.rnn.decoders import HierarchicalRNNDecoder, RNNAutoregressiveDecoder
from resolv_ml.models.seq2seq.rnn.encoders import BidirectionalRNNEncoder
from resolv_ml.training.trainer import Trainer
from resolv_pipelines.data.loaders import TFRecordLoader
from resolv_pipelines.data.representation.mir import PitchSequenceRepresentation

from scripts.utilities.constants import Paths


def get_hierarchical_model(attribute_reg_layer: AttributeRegularizationLayer):
    model_config_path = Paths.ML_CONFIG_DIR / "model.json"
    with open(model_config_path) as file:
        model_config = json.load(file)

    encoder_config = model_config["encoder"]
    encoder = BidirectionalRNNEncoder(
        enc_rnn_sizes=encoder_config["enc_rnn_sizes"],
        embedding_layer=keras.layers.Embedding(
            input_dim=model_config["num_classes"],
            output_dim=model_config["embedding_size"],
            name="encoder_embedding"
        ),
        dropout=encoder_config["dropout"]
    )

    hier_decoder_config = model_config["hier_decoder"]
    core_decoder_config = hier_decoder_config["core_decoder"]
    decoder = HierarchicalRNNDecoder(
        level_lengths=hier_decoder_config["level_lengths"],
        dec_rnn_sizes=hier_decoder_config["dec_rnn_sizes"],
        dropout=hier_decoder_config["dropout"],
        core_decoder=RNNAutoregressiveDecoder(
            dec_rnn_sizes=core_decoder_config["dec_rnn_sizes"],
            num_classes=model_config["num_classes"],
            embedding_layer=keras.layers.Embedding(
                input_dim=model_config["num_classes"],
                output_dim=model_config["embedding_size"],
                name="decoder_embedding"
            ),
            dropout=core_decoder_config["dropout"],
            sampling_schedule=core_decoder_config["sampling_schedule"],
            sampling_rate=core_decoder_config["sampling_rate"]
        )
    )

    model = AttributeRegularizedVAE(
        z_size=model_config["z_size"],
        input_processing_layer=encoder,
        generative_layer=decoder,
        attribute_regularization_layer=attribute_reg_layer,
        max_beta=model_config["max_beta"],
        beta_rate=model_config["beta_rate"],
        free_bits=model_config["free_bits"]
    )
    return model, model_config


def load_datasets(train_dataset_path: str, val_dataset_path: int, attribute: str):
    train_data = load_pitch_seq_dataset(dataset_path=train_dataset_path, attribute=attribute)
    val_data = load_pitch_seq_dataset(dataset_path=val_dataset_path, attribute=attribute)
    return train_data, val_data


def load_pitch_seq_dataset(dataset_path: str, attribute: str) -> tf.data.TFRecordDataset:

    def map_fn(ctx, seq):
        input_seq = tf.transpose(seq["pitch_seq"])
        attributes = ctx[attribute]
        target = input_seq
        return (input_seq, attributes), target

    dataset_config_path = Paths.ML_CONFIG_DIR / "dataset.json"
    with open(dataset_config_path) as file:
        dataset_config = json.load(file)

    trainer_config_path = Paths.ML_CONFIG_DIR / "trainer.json"
    with open(trainer_config_path) as file:
        trainer_config = json.load(file)

    representation = PitchSequenceRepresentation(sequence_length=dataset_config["sequence_length"])
    tfrecord_loader = TFRecordLoader(
        file_pattern=dataset_path,
        parse_fn=functools.partial(
            representation.parse_example,
            parse_sequence_feature=True,
            attributes_to_parse=[attribute]
        ),
        map_fn=map_fn,
        batch_size=trainer_config["fit"]["batch_size"],
        batch_drop_reminder=True,
        shuffle=dataset_config["shuffle"],
        shuffle_buffer_size=dataset_config["shuffle_buffer_size"],
        shuffle_repeat=dataset_config["shuffle_repeat"],
        cache_dataset=dataset_config["cache_dataset"],
        cache_filename=dataset_config["cache_filename"],
        prefetch_buffer_size=dataset_config["prefetch_buffer_size"],
        interleave_cycle_length=dataset_config["interleave_cycle_length"],
        interleave_block_length=dataset_config["interleave_block_length"],
        num_parallel_calls=dataset_config["num_parallel_calls"],
        deterministic=dataset_config["deterministic"],
        seed=dataset_config["seed"]
    )
    return tfrecord_loader.load_dataset()


def get_trainer(model: keras.Model) -> Trainer:
    trainer_config_path = Paths.ML_CONFIG_DIR / "trainer.json"
    trainer = Trainer(model, config_file_path=trainer_config_path)
    return trainer
