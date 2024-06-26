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
from resolv_ml.utilities.distributions.power_transforms import BoxCox, YeoJohnson
from scipy.stats import boxcox, yeojohnson, kurtosis

import utilities


def test_power_transforms(args):
    class PowerTransformModel(keras.Model):

        def __init__(self, pt_id: str, lambd: int):
            super(PowerTransformModel, self).__init__()
            if pt_id == 'box-cox':
                self._pt_layer = BoxCox(lambda_init=lambd, trainable=False)
            elif pt_id == 'yeo-johnson':
                self._pt_layer = YeoJohnson(lambda_init=lambd, trainable=False)
            else:
                raise ValueError(f'Unknown pt_id: {pt_id}')
            self.name = self._pt_layer.name

        def call(self, inputs):
            return self._pt_layer(inputs)

    for attribute in args.attributes:
        for power_transform_id in args.power_transform_ids:
            logging.info(f"Evaluating {power_transform_id} power transform for attribute '{attribute}'...")
            output_path = Path(args.histogram_output_path) / "power-transforms" / power_transform_id.lower() / attribute
            if not output_path.exists():
                numpy_output_path = output_path / "numpy"
                output_path.mkdir(parents=True, exist_ok=True)
                numpy_output_path.mkdir(parents=True, exist_ok=True)
                _, llm_lmbda = estimate_llm_pt_lambda(args, attribute, power_transform_id)
                logging.info(f"LLM power transform {power_transform_id} best lambda value for attribute '{attribute}'"
                             f" is: {llm_lmbda:.5f}'")
                # Create PowerTransform model
                pt_model = PowerTransformModel(power_transform_id, llm_lmbda)
                pt_model.trainable = False
                pt_model.compile()
                # Load dataset
                dataset = utilities.load_dataset(dataset_path=args.dataset_path,
                                                 sequence_length=args.sequence_length,
                                                 attribute=attribute,
                                                 batch_size=args.batch_size,
                                                 parse_sequence_feature=False)
                # Compute PowerTransform
                pt_out = pt_model.predict(dataset)
                pt_out_norm = (pt_out - k_ops.mean(pt_out)) / k_ops.std(pt_out)
                pt_out_norm = pt_out_norm.numpy()
                # Compute Kurtosis
                kurt = kurtosis(pt_out_norm)[0]
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
                    power_transform_id=power_transform_id,
                    lmbda=llm_lmbda,
                    attribute=attribute,
                    histogram_bins=args.histogram_bins
                )
                # Save output distributions
                logging.info(f"Saving output distributions to numpy file...")
                numpy_pt_out_filename = (f'pt_out_norm_{power_transform_id.lower()}_{attribute}_lambda_'
                                         f',{llm_lmbda:.2f}.npy')
                np.save(numpy_output_path / numpy_pt_out_filename, pt_out_norm)
            else:
                logging.info(f"Power transform distribution for attribute '{attribute}' already exists."
                             f"Remove the folder {output_path} to override it.")


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


def estimate_llm_pt_lambda(args, attribute: str, power_transform_id: str):
    attribute_data = utilities.load_flat_dataset(dataset_path=args.dataset_path,
                                                 sequence_length=args.sequence_length,
                                                 attribute=attribute,
                                                 batch_size=args.batch_size,
                                                 parse_sequence_feature=False)
    attribute_data = attribute_data.squeeze()
    if power_transform_id == 'box-cox':
        y, lambda_llf = boxcox(attribute_data, lmbda=None)
    elif power_transform_id == 'yeo-johnson':
        y, lambda_llf = yeojohnson(attribute_data, lmbda=None)
    else:
        raise ValueError(f'Unknown pt_id: {power_transform_id}')
    return y, lambda_llf


def negentropy_approx_naive(x):
    return (1 / 12) * np.mean(x ** 3) ** 2 + (1 / 48) * kurtosis(x)[0] ** 2


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
                          power_transform_id: str,
                          lmbda: float,
                          attribute: str,
                          histogram_bins: List[int]):
    histograms_output_path = output_path / "histograms"
    histograms_output_path.mkdir(parents=True, exist_ok=True)
    pt_title = power_transform_id
    attr_title = attribute.replace("_", " ").capitalize()
    for bins in histogram_bins:
        filename = (f'{str(histograms_output_path)}/histogram_{power_transform_id.lower()}_{attribute}'
                    f'_lambda_{lmbda:.2f}_bins_{bins}.png')
        plt.hist(data, bins=bins, color='blue', alpha=0.7)
        plt.suptitle(f'{pt_title} - {attr_title}')
        plt.title(r'$\lambda$ = ' + f'{lmbda:.2f} - Bins = {bins}')
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
    parser.add_argument('--seed', help='Seed for random initializers.', required=False, type=int)
    parser.add_argument('--logging-level', help='Set the logging level.', default="INFO", required=False,
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO"])
    os.environ["KERAS_BACKEND"] = "tensorflow"
    vargs = parser.parse_args()
    if vargs.seed:
        keras.utils.set_random_seed(vargs.seed)
        np.random.seed(vargs.seed)
        tf.config.experimental.enable_op_determinism()
    logging.getLogger().setLevel(vargs.logging_level)
    test_original_distributions(vargs)
    test_power_transforms(vargs)
