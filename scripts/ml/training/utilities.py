import argparse
import functools
import json
import logging
from pathlib import Path
from typing import List

import keras
import tensorflow as tf
from resolv_ml.models.dlvm.vae.ar_vae import AttributeRegularizedVAE, AttributeRegularizationLayer
from resolv_ml.models.seq2seq.rnn.decoders import HierarchicalRNNDecoder, RNNAutoregressiveDecoder
from resolv_ml.models.seq2seq.rnn.encoders import BidirectionalRNNEncoder
from resolv_ml.training.trainer import Trainer
from resolv_pipelines.data.loaders import TFRecordLoader
from resolv_pipelines.data.representation.mir import PitchSequenceRepresentation


def check_tf_gpu_availability():
    gpu_list = tf.config.list_physical_devices('GPU')
    if len(gpu_list) > 0:
        logging.info(f'Num GPUs Available: {len(gpu_list)}. List: {gpu_list}')
    return gpu_list


def get_distributed_strategy(gpu_ids: List[int] = None) -> tf.distribute.Strategy:
    gpu_list = check_tf_gpu_availability()

    if not gpu_list:
        raise SystemExit("GPU not available.")

    if gpu_ids and any(gpu_id >= len(gpu_list) for gpu_id in gpu_ids):
        raise ValueError(f"GPU ids {gpu_ids} not valid. There are only {len(gpu_ids)} GPUs available.")

    if gpu_ids:
        selected_gpus = [gpu_list[gpu] for gpu in gpu_ids]
    else:
        logging.info(f"No GPU ids provided. Using default GPU device {gpu_list[0]}.")
        selected_gpus = [gpu_list[0]]
    selected_gpus_name = [selected_gpu.name.replace("/physical_device:", "") for selected_gpu in selected_gpus]

    if len(selected_gpus) > 1:
        strategy = tf.distribute.MirroredStrategy(devices=selected_gpus_name)
    else:
        strategy = tf.distribute.OneDeviceStrategy(device=selected_gpus_name[0])
    return strategy


def get_hierarchical_model(model_config_path: Path,
                           attribute_reg_layer: AttributeRegularizationLayer) -> AttributeRegularizedVAE:
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
        max_beta=model_config["hparams"]["max_beta"],
        beta_rate=model_config["hparams"]["beta_rate"],
        free_bits=model_config["hparams"]["free_bits"]
    )
    return model


def load_datasets(dataset_config_path: Path,
                  trainer_config_path: Path,
                  train_dataset_path: str,
                  val_dataset_path: int,
                  attribute: str):
    train_data, input_shape = load_pitch_seq_dataset(dataset_config_path=dataset_config_path,
                                                     trainer_config_path=trainer_config_path,
                                                     dataset_path=train_dataset_path,
                                                     attribute=attribute)
    val_data, _ = load_pitch_seq_dataset(dataset_config_path=dataset_config_path,
                                         trainer_config_path=trainer_config_path,
                                         dataset_path=val_dataset_path,
                                         attribute=attribute)
    return train_data, val_data, input_shape


def load_pitch_seq_dataset(dataset_config_path: Path,
                           trainer_config_path: Path,
                           dataset_path: str,
                           attribute: str) -> tf.data.TFRecordDataset:
    def get_input_shape():
        input_seq_shape = batch_size, dataset_config["sequence_length"], dataset_config["sequence_features"]
        aux_input_shape = (batch_size,)
        return input_seq_shape, aux_input_shape

    def map_fn(ctx, seq):
        input_seq = tf.transpose(seq["pitch_seq"])
        attributes = ctx[attribute]
        target = input_seq
        return (input_seq, attributes), target

    with open(dataset_config_path) as file:
        dataset_config = json.load(file)

    with open(trainer_config_path) as file:
        trainer_config = json.load(file)

    batch_size = trainer_config["fit"]["batch_size"]
    representation = PitchSequenceRepresentation(sequence_length=dataset_config["sequence_length"])
    tfrecord_loader = TFRecordLoader(
        file_pattern=dataset_path,
        parse_fn=functools.partial(
            representation.parse_example,
            parse_sequence_feature=True,
            attributes_to_parse=[attribute]
        ),
        map_fn=map_fn,
        batch_size=batch_size,
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
    return tfrecord_loader.load_dataset(), get_input_shape()


def get_trainer(trainer_config_path: Path, model: keras.Model) -> Trainer:
    trainer = Trainer(model, config_file_path=trainer_config_path)
    trainer.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(), keras.metrics.SparseTopKCategoricalAccuracy()]
    )
    return trainer


def get_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model-config-path', help='Path to the model\'s configuration file.', required=True)
    parser.add_argument('--trainer-config-path', help='Path to the trainer\'s configuration file.', required=True)
    parser.add_argument('--dataset-config-path', help='Path to the dataset\'s configuration file.', required=True)
    parser.add_argument('--train-dataset-path', help='Path to training dataset.', required=True)
    parser.add_argument('--val-dataset-path', help='Path to validation dataset.', required=True)
    parser.add_argument('--attribute', help='Attribute to regularize.', required=True)
    parser.add_argument('--reg-dim', help='Latent code regularization dimension.', default=0, type=int)
    parser.add_argument('--gamma', help='Gamma factor to scale regularization loss.', default=1.0, type=float)
    parser.add_argument('--gpus', nargs="+", help='ID of GPUs to use for training.', required=False,
                        default=[], type=int)
    parser.add_argument('--logging-level', help='Set the logging level.', default="INFO", required=False,
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO"])
    return parser
