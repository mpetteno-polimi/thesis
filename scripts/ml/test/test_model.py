import argparse
import logging
import os
from pathlib import Path

import keras
import matplotlib.pyplot as plt
from resolv_mir.note_sequence.attributes import compute_attribute
from resolv_pipelines.data.representation.mir import PitchSequenceRepresentation

import utilities

import uuid
from resolv_mir.note_sequence.io import midi_io


def test_model_regularization(args):
    Path(args.plot_output_path).mkdir(parents=True, exist_ok=True)
    for attribute in args.attributes:
        dataset = utilities.load_dataset(dataset_path=args.dataset_path,
                                         sequence_length=args.sequence_length,
                                         attribute=attribute,
                                         batch_size=args.batch_size,
                                         parse_sequence_feature=True)
        model = keras.saving.load_model(args.model_path, compile=False)
        input_sequences = []
        input_sequences_attributes = []
        latent_codes = []
        z_mean = []
        z_var = []
        z_reg_dimension = []
        z_non_reg_dimension = []
        decoded_sequences = []

        representation = PitchSequenceRepresentation(args.sequence_length)

        for batch in dataset:
            batch_input_sequence, batch_sequences_attributes = batch
            input_sequences.append(batch_input_sequence)
            input_sequences_attributes.append(batch_sequences_attributes)
            batch_latent_codes, batch_z_mean, batch_z_var = model.encode(batch)
            latent_codes.append(batch_latent_codes)
            z_mean.append(batch_z_mean)
            z_var.append(batch_z_var)
            z_reg_dimension.append(batch_latent_codes[:, args.regularized_dimension])
            z_non_reg_dimension.append(batch_latent_codes[:, args.non_regularized_dimension])
            batch_decoded_sequences = model.decode(batch_latent_codes)
            decoded_sequences.append(batch_decoded_sequences)

        input_sequences = keras.ops.concatenate(input_sequences, axis=0)
        input_sequences_attributes = keras.ops.concatenate(input_sequences_attributes, axis=0)
        latent_codes = keras.ops.concatenate(latent_codes, axis=0)
        z_mean = keras.ops.concatenate(z_mean, axis=0)
        z_var = keras.ops.concatenate(z_var, axis=0)
        z_reg_dimension = keras.ops.concatenate(z_reg_dimension, axis=0)
        z_non_reg_dimension = keras.ops.concatenate(z_non_reg_dimension, axis=0)
        decoded_sequences = keras.ops.concatenate(decoded_sequences, axis=0)

        for input_sequence in input_sequences:
            input_sequence = keras.ops.squeeze(input_sequence, axis=-1)
            ns = representation.to_canonical_format(input_sequence.numpy(), attributes=None)
            midi_io.note_sequence_to_midi_file(ns, f"./output/tmp/{uuid.uuid4()}.midi")

        regularization_scatter_plot(
            output_path=f'{args.plot_output_path}/plot1.png',
            title="",
            reg_dim_data=z_reg_dimension,
            reg_dim_idx=args.regularized_dimension,
            non_reg_dim_data=z_non_reg_dimension,
            non_reg_dim_idx=args.non_regularized_dimension,
            attributes=input_sequences_attributes,
            attribute_name=attribute,
            colorbar=True
        )

        regularization_scatter_plot(
            output_path=f'{args.plot_output_path}/plot2.png',
            title="",
            reg_dim_data=z_reg_dimension,
            reg_dim_idx=args.regularized_dimension,
            non_reg_dim_data=z_non_reg_dimension,
            non_reg_dim_idx=args.non_regularized_dimension,
            attributes=compute_sequences_attributes(decoded_sequences, attribute, args.sequence_length),
            attribute_name=attribute,
            colorbar=True
        )


def test_model_generation(args):
    model = keras.saving.load_model(args.model_path)
    for attribute in args.attributes:
        # sample N instances from N(0, I)
        latent_codes = keras.random.normal(shape=(1024, 256), seed=42)
        # control regularized dimension
        # latent_codes = []
        # generate the sequence
        generated_sequences = model.predict(latent_codes, batch_size=args.batch_size)
        pass
        # generated_sequences_attributes = compute_sequences_attributes(generated_sequences, attribute, args.sequence_length)
        # compute Pearson coefficient

        # plot regularized dimension vs decoded sequences attributes and fit linear model


def compute_sequences_attributes(decoded_sequences, attribute_name: str, sequence_length: int):
    decoded_ns_attributes = []
    representation = PitchSequenceRepresentation(sequence_length)
    for decoded_sequence in decoded_sequences:
        decoded_note_sequence = representation.to_canonical_format(decoded_sequence.numpy(), attributes=None)
        decoded_ns_attribute = compute_attribute(decoded_note_sequence, attribute_name)
        decoded_ns_attributes.append(decoded_ns_attribute)
    return decoded_ns_attributes


def regularization_scatter_plot(output_path: str,
                                title: str,
                                reg_dim_data,
                                reg_dim_idx,
                                non_reg_dim_data,
                                non_reg_dim_idx,
                                attributes,
                                attribute_name: str,
                                colorbar: bool = True):
    plt.scatter(x=non_reg_dim_data, y=reg_dim_data, c=attributes, cmap='viridis', alpha=0.8)
    if colorbar:
        plt.colorbar(label=attribute_name)
    plt.xlabel(f'z_{non_reg_dim_idx}')
    plt.ylabel(f'z_{reg_dim_idx}')
    plt.title(title)
    plt.savefig(output_path, format='png', dpi=300)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--model-path', required=True, help='Path to the dataset containing SequenceExample with the '
                                                            'attributes to evaluate saved in the context.')
    parser.add_argument('--dataset-path', required=True, help='Path to the dataset containing the test SequenceExample')
    parser.add_argument('--sequence-length', help='Length of the sequences in the dataset.', required=True, type=int)
    parser.add_argument('--attributes', nargs='+', help='Attributes to evaluate.', required=True)
    parser.add_argument('--regularized-dimension', help='Index of the latent code regularized dimension.',
                        required=False, type=int, default=0)
    parser.add_argument('--non-regularized-dimension', help='Index of the latent code non regularized dimension.',
                        required=False, type=int, default=127)
    parser.add_argument('--plot-output-path', help='Path where the histograms will be saved.', required=True)
    parser.add_argument('--batch-size', help='Batch size.', required=False, default=64, type=int,
                        choices=[32, 64, 128, 256, 512])
    parser.add_argument('--logging-level', help='Set the logging level.', default="INFO", required=False,
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO"])
    os.environ["KERAS_BACKEND"] = "tensorflow"
    vargs = parser.parse_args()
    logging.getLogger().setLevel(vargs.logging_level)
    # test_model_regularization(vargs)
    test_model_generation(vargs)
