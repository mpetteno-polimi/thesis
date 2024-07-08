import argparse
import logging
import os
from pathlib import Path
from typing import List, Callable

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import ops as k_ops
from resolv_ml.utilities.bijectors.power_transform import BoxCox
from resolv_mir.note_sequence.attributes import ATTRIBUTE_FN_MAP
from scipy.stats import boxcox, kurtosis

import utilities


def test_power_transform(args):
    for attribute in args.attributes:
        logging.info(f"Evaluating power transform for attribute '{attribute}'...")
        main_output_path = Path(args.histogram_output_path) / "power-transform" / attribute
        if not main_output_path.exists():
            # Load dataset
            dataset = utilities.load_flat_dataset(dataset_path=args.dataset_path,
                                                  sequence_length=args.sequence_length,
                                                  attribute=attribute,
                                                  batch_size=args.batch_size,
                                                  parse_sequence_feature=False)[:, 0]
            shifts_grid = [0.] if not args.shift_grid_search \
                else keras.ops.arange(args.shift_min, args.shift_max, args.shift_step)
            for s in shifts_grid:
                output_path = main_output_path / f"shift_{s:.2f}"
                output_path.mkdir(parents=True, exist_ok=True)
                numpy_output_path = output_path / "numpy"
                numpy_output_path.mkdir(parents=True, exist_ok=True)
                # add epsilon to shift in order to avoid zero input values for BoxCox
                shifted_inputs = dataset + s + 1e-5
                # Compute best power parameter for BoxCox using log-likelihood maximization
                _, llm_lmbda = boxcox(shifted_inputs, lmbda=None)
                logging.info(f"LLM power transform best power value for attribute '{attribute}' shifted by {s:.2f} "
                             f"is: {llm_lmbda:.5f}'")
                # Create PowerTransform bijector
                power_transform_bij = BoxCox(
                    power=llm_lmbda,
                    shift=0.,
                    power_trainable=False,
                    shift_trainable=False
                )
                # Compute PowerTransform
                pt_out = power_transform_bij.inverse(shifted_inputs)
                pt_out_norm = (pt_out - k_ops.mean(pt_out)) / k_ops.std(pt_out)
                pt_out_norm = pt_out_norm.numpy()
                # Compute Kurtosis
                kurt = kurtosis(pt_out_norm)
                logging.info(f"Kurtosis index is {kurt:.5f}.")
                # Compute Negentropy Naive
                negentropy_naive = negentropy_approx_naive(pt_out_norm)
                logging.info(f"Negentropy naive index is {negentropy_naive:.5f}.")
                # Compute Negentropy exp
                negentropy_exp = negentropy_approx_fn(pt_out_norm, lambda u: -np.exp(-u ** 2 / 2))
                logging.info(f"Negentropy exp index is {negentropy_exp:.5f}.")
                # Compute Negentropy cosh
                negentropy_cosh = negentropy_approx_fn(pt_out_norm, lambda u: np.log(np.cosh(u)))
                logging.info(f"Negentropy cosh index is {negentropy_cosh:.5f}.")
                # Plot histogram of the output distribution
                logging.info(f"Plotting output distribution histogram...")
                plot_pt_distributions(
                    data=pt_out_norm,
                    output_path=output_path,
                    power=llm_lmbda,
                    shift=s,
                    attribute=attribute,
                    histogram_bins=args.histogram_bins
                )
                # Save output distributions
                logging.info(f"Saving output distributions to numpy file...")
                numpy_pt_out_filename = f'pt_out_norm_{attribute}_power_{llm_lmbda:.2f}_shift_{s:.2f}.npy'
                np.save(numpy_output_path / numpy_pt_out_filename, pt_out_norm)
        else:
            logging.info(f"Power transform distribution for attribute '{attribute}' already exists."
                         f"Remove the folder {main_output_path} to override it.")


def test_original_distributions(args):
    for attribute in args.attributes:
        logging.info(f"Evaluating original distribution for attribute '{attribute}'...")
        output_path = Path(args.histogram_output_path) / "original" / attribute
        if not output_path.exists():
            attribute_data = utilities.load_flat_dataset(dataset_path=args.dataset_path,
                                                         sequence_length=args.sequence_length,
                                                         attribute=attribute,
                                                         batch_size=args.batch_size,
                                                         parse_sequence_feature=False)
            zero_count = np.count_nonzero(np.where(np.isclose(attribute_data, 0)))
            logging.info(f"Original zero elements: {zero_count}/{len(attribute_data)}")
            plot_original_distributions(attribute_data, output_path, attribute, args.histogram_bins)
        else:
            logging.info(f"Original distribution for attribute '{attribute}' already exists."
                         f"Remove the folder {output_path} to override it.")


def negentropy_approx_naive(x):
    return (1 / 12) * np.mean(x ** 3) ** 2 + (1 / 48) * kurtosis(x) ** 2


def negentropy_approx_fn(x, fn: Callable):
    gaussian_data = np.random.normal(0, 1, x.shape[0])
    negentropy = (np.mean(fn(x)) - np.mean(fn(gaussian_data))) ** 2
    return negentropy


def plot_original_distributions(data,
                                output_path: Path,
                                attribute: str,
                                histogram_bins: List[int]):
    histograms_output_path = output_path / "histograms"
    histograms_output_path.mkdir(parents=True, exist_ok=True)
    for bins in histogram_bins:
        filename = f'{str(histograms_output_path)}/histogram_original_{attribute}_{bins}_bins.png'
        logging.info(f"Plotting original histogram with {bins} bins for attribute {attribute}...")
        plt.hist(data, bins=bins, color='blue', alpha=0.7)
        plt.title(f'{attribute.replace("_", " ").capitalize()} - {bins} bins')
        plt.grid(linestyle=':')
        plt.savefig(filename, format='png', dpi=300)
        plt.close()


def plot_pt_distributions(data,
                          output_path: Path,
                          power: float,
                          shift: float,
                          attribute: str,
                          histogram_bins: List[int]):
    histograms_output_path = output_path / "histograms"
    histograms_output_path.mkdir(parents=True, exist_ok=True)
    attr_title = attribute.replace("_", " ").capitalize()
    for bins in histogram_bins:
        filename = (f'{str(histograms_output_path)}/histogram_{attribute}_power_{power:.2f}_shift_{shift:.2f}'
                    f'_bins_{bins}.png')
        plt.hist(data, bins=bins, color='blue', alpha=0.7)
        plt.suptitle(f'BoxCox - {attr_title}')
        plt.title(r'$\lambda$ = ' + f'{power:.2f} - {shift:.2f} - Bins = {bins}')
        plt.grid(linestyle=':')
        plt.savefig(filename, format='png', dpi=300)
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate power transforms for specified attributes contained in the given SequenceExample "
                    "dataset. Supported transformations are BoxCox and YeoJohnson."
    )
    parser.add_argument('--dataset-path', required=True, help='Path to the dataset containing SequenceExample with the '
                                                              'attributes to evaluate saved in the context.')
    parser.add_argument('--sequence-length', help='Length of the sequences in the dataset.', required=True, type=int)
    parser.add_argument('--attributes', nargs='+', help='Attributes to evaluate.', required=True)
    parser.add_argument('--histogram-output-path', help='Path where the histograms will be saved.', required=True)
    parser.add_argument('--histogram-bins', help='Number of bins for the histogram.', nargs='+', default=[60],
                        required=False, type=int)
    parser.add_argument('--batch-size', help='Batch size.', required=False, default=64, type=int,
                        choices=[32, 64, 128, 256, 512])
    parser.add_argument('--power-transform-ids', help="IDs of power transforms to evaluate.", nargs='+',
                        choices=["box-cox", "yeo-johnson"], default=["box-cox", "yeo-johnson"], required=False)
    parser.add_argument('--shift-grid-search', help='Do a grid search for the power transform shift parameter.',
                        action="store_true")
    parser.add_argument('--shift-min', help='Start value for the grid search range of the shift parameter for the '
                                            'BoxCox power transform.',
                        default=0., required=False, type=float)
    parser.add_argument('--shift-max', help='Stop value for the grid search range of the shift parameter for the '
                                            'BoxCox power transform.',
                        default=3., required=False, type=float)
    parser.add_argument('--shift-step', help='Increment step for the grid search range of the shift parameter for the '
                                             'BoxCox power transform.',
                        default=0.25, required=False, type=float)
    parser.add_argument('--seed', help='Seed for random initializers.', required=False, type=int)
    parser.add_argument('--logging-level', help='Set the logging level.', default="INFO", required=False,
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO"])
    os.environ["KERAS_BACKEND"] = "tensorflow"
    vargs = parser.parse_args()
    if vargs.attributes[0] == "all":
        vargs.attributes = ATTRIBUTE_FN_MAP.keys()
    if vargs.seed:
        keras.utils.set_random_seed(vargs.seed)
        np.random.seed(vargs.seed)
        tf.config.experimental.enable_op_determinism()
    logging.getLogger().setLevel(vargs.logging_level)
    test_original_distributions(vargs)
    test_power_transform(vargs)
