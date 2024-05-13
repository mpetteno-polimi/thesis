import logging
import os
from pathlib import Path

import keras
import matplotlib.pyplot as plt

import utilities


def test_model_regularization(args):
    output_dir = Path(args.output_path) / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    for attribute in args.attributes:
        dataset = utilities.load_dataset(dataset_path=args.test_dataset_path,
                                         sequence_length=args.sequence_length,
                                         attribute=attribute,
                                         batch_size=args.batch_size,
                                         parse_sequence_feature=True)
        model = keras.saving.load_model(args.model_path, compile=False)
        model.compile(run_eagerly=True)
        decoded_sequences, latent_codes, input_sequences, input_sequences_attributes = (
            model.predict(dataset, steps=args.dataset_cardinality//args.batch_size))

        regularization_scatter_plot(
            output_path=str(output_dir/"encoded_sequences_reg_latent_space.png"),
            title="Latent distribution of encoded sequences",
            reg_dim_data=latent_codes[:, args.regularized_dimension],
            reg_dim_idx=args.regularized_dimension,
            non_reg_dim_data=latent_codes[:, args.non_regularized_dimension],
            non_reg_dim_idx=args.non_regularized_dimension,
            attributes=input_sequences_attributes,
            attribute_name=attribute,
            colorbar=True
        )

        decoded_sequences_attrs, hold_note_start_seq_count = utilities.compute_sequences_attributes(
            decoded_sequences, attribute, args.sequence_length)
        regularization_scatter_plot(
            output_path=str(output_dir/"decoded_sequences_reg_latent_space.png"),
            title="Latent distribution of generated sequences",
            reg_dim_data=latent_codes[:, args.regularized_dimension],
            reg_dim_idx=args.regularized_dimension,
            non_reg_dim_data=latent_codes[:, args.non_regularized_dimension],
            non_reg_dim_idx=args.non_regularized_dimension,
            attributes=decoded_sequences_attrs,
            attribute_name=attribute,
            colorbar=True
        )

        logging.info(f"Decoded {hold_note_start_seq_count} sequences that start with an hold note token "
                     f"({hold_note_start_seq_count*100/args.dataset_cardinality:.2f}%).")


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
    plt.xlabel(f'$z_{{{non_reg_dim_idx}}}$')
    plt.ylabel(f'$z_{{{reg_dim_idx}}}$')
    plt.title(title)
    plt.savefig(output_path, format='png', dpi=300)
    plt.close()


if __name__ == '__main__':
    parser = utilities.get_arg_parser("")
    os.environ["KERAS_BACKEND"] = "tensorflow"
    vargs = parser.parse_args()
    logging.getLogger().setLevel(vargs.logging_level)
    test_model_regularization(vargs)
