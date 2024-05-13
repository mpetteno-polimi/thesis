import logging
import os
import random
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
from resolv_mir.note_sequence.io import midi_io
from resolv_pipelines.data.representation.mir import PitchSequenceRepresentation

import utilities


def test_model_generation(args):
    for attribute in args.attributes:
        model = keras.saving.load_model(args.model_path, compile=False)
        model.compile(run_eagerly=True)
        # sample N = dataset_cardinality instances from model's prior
        latent_codes = model.sample(keras.ops.convert_to_tensor(args.dataset_cardinality)).numpy()
        # control regularized dimension
        if args.control_reg_dim:
            # load the test dataset and extract all the attribute values to get min and max
            dataset = utilities.load_dataset(dataset_path=args.test_dataset_path,
                                             sequence_length=args.sequence_length,
                                             attribute=attribute,
                                             batch_size=args.batch_size,
                                             parse_sequence_feature=False)
            attribute_values = np.array([x for batch in dataset for x in batch])
            attr_min_val = np.min(attribute_values)
            attr_max_val = np.max(attribute_values)
            latent_codes[:, args.regularized_dimension] = keras.ops.linspace(start=attr_min_val, stop=attr_max_val,
                                                                             num=args.dataset_cardinality)
        # generate the sequence
        generated_sequences = model.decode(inputs=(latent_codes, keras.ops.convert_to_tensor(args.sequence_length)))
        generated_sequences_attrs, hold_note_start_seq_count = utilities.compute_sequences_attributes(
            generated_sequences.numpy(), attribute, args.sequence_length)
        # compute Pearson coefficient and plot graph with the best linear fitting model
        reg_dim_data = latent_codes[:, args.regularized_dimension]
        correlation_matrix = np.corrcoef(reg_dim_data, generated_sequences_attrs)
        pearson_coefficient = correlation_matrix[0, 1]
        output_dir = Path(args.output_path) / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        slope, intercept = plot_reg_dim_vs_attribute(output_path=str(output_dir / 'reg_dim_vs_attribute.png'),
                                                     title="",
                                                     reg_dim_data=reg_dim_data,
                                                     reg_dim_idx=args.regularized_dimension,
                                                     attribute_data=generated_sequences_attrs,
                                                     attribute_name=attribute)
        # convert generated sequences to MIDI and save to disk
        representation = PitchSequenceRepresentation(args.sequence_length)
        seq_to_save_count = min(args.dataset_cardinality, args.num_midi_to_save)
        random_idxes = [random.randint(0, args.dataset_cardinality) for _ in range(seq_to_save_count)]
        for idx, generated_sequence in enumerate(keras.ops.take(generated_sequences, indices=random_idxes, axis=0)):
            generated_note_sequence = representation.to_canonical_format(generated_sequence, attributes=None)
            filename = f"midi/{attribute}_{latent_codes[random_idxes[idx], args.regularized_dimension]:.2f}.midi"
            midi_io.note_sequence_to_midi_file(generated_note_sequence, Path(args.output_path)/filename)

        logging.info(f"Best linear model fit parameters. Slope: {slope:.2f}, Intercept: {intercept:.2f}")
        logging.info(f"Pearson coefficient {pearson_coefficient:.2f}.")
        logging.info(f"Generated {hold_note_start_seq_count} sequences that start with an hold note token "
                     f"({hold_note_start_seq_count * 100 / args.dataset_cardinality:.2f}%).")


def plot_reg_dim_vs_attribute(output_path: str,
                              title: str,
                              reg_dim_data,
                              reg_dim_idx,
                              attribute_data,
                              attribute_name: str):
    slope, intercept = np.polyfit(reg_dim_data, attribute_data, 1)
    plt.scatter(reg_dim_data, attribute_data, color='blue', s=5)
    plt.plot(reg_dim_data, slope * reg_dim_data + intercept, color='red')
    plt.ylabel(f'{attribute_name}')
    plt.xlabel(f'$z_{reg_dim_idx}$')
    plt.title(title)
    plt.savefig(output_path, format='png', dpi=300)
    plt.close()
    return slope, intercept


if __name__ == '__main__':
    parser = utilities.get_arg_parser("")
    parser.add_argument('--control-reg-dim', action="store_true",
                        help='Control the regularized dimension of the sampled latent codes using the min and max '
                             'values for the given attribute in the given test dataset.')
    parser.add_argument('--num-midi-to-save', help='Number of generated sequences to save as MIDI file. '
                                                   'The N sequences will be chosen randomly.', required=True, type=int)
    os.environ["KERAS_BACKEND"] = "tensorflow"
    vargs = parser.parse_args()
    logging.getLogger().setLevel(vargs.logging_level)
    test_model_generation(vargs)
