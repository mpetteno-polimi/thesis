import argparse
import logging
import os
from pathlib import Path
from typing import List

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import ops as k_ops
from resolv_ml.utilities.distributions.power_transforms import BoxCox, YeoJohnson

import utilities


def test_power_transforms(args):
    class PowerTransformModel(keras.Model):

        def __init__(self, pt_id: str, lambd: int):
            super(PowerTransformModel, self).__init__()
            if pt_id == 'box-cox':
                self._pt_layer = BoxCox(lambda_init=lambd)
            elif pt_id == 'yeo-johnson':
                self._pt_layer = YeoJohnson(lambda_init=lambd)
            else:
                raise ValueError(f'Unknown pt_id: {pt_id}')
            self.name = self._pt_layer.name

        def call(self, inputs):
            return self._pt_layer(inputs)

    for attribute in args.attributes:
        for idx, power_transform_id in enumerate(args.power_transform_ids):
            output_path = Path(args.histogram_output_path) / "power-transforms" / power_transform_id.lower() / attribute
            output_path.mkdir(parents=True, exist_ok=True)
            numpy_output_path = output_path / "numpy"
            numpy_output_path.mkdir(parents=True, exist_ok=True)
            kl_divs = []
            # Compute lambda range
            lambda_min = args.lambda_min[idx] if len(args.lambda_min) > 1 else args.lambda_min[0]
            lambda_max = args.lambda_max[idx] if len(args.lambda_max) > 1 else args.lambda_max[0]
            lambda_step = args.lambda_step[idx] if len(args.lambda_step) > 1 else args.lambda_step[0]
            lambda_range = k_ops.arange(lambda_min, lambda_max, lambda_step)
            for lmbda in lambda_range:
                logging.info(f"Evaluating {power_transform_id} power transform with lambda {lmbda:.2f} for attribute "
                             f"'{attribute}'...")
                # Create PowerTransform model
                pt_model = PowerTransformModel(power_transform_id, lmbda)
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
                pt_out_mean = k_ops.mean(pt_out)
                pt_out_std = k_ops.std(pt_out)
                # Compute KLD
                logging.info(f"Computing Kullback–Leibler Divergence...")
                kl_div = -0.5 * k_ops.sum((1 + k_ops.log(pt_out_std) - k_ops.square(pt_out_mean) - pt_out_std))
                kl_divs.append(kl_div)
                logging.info(f"Kullback–Leibler Divergence is {kl_div:.2f}.")
                # Plot histogram of the output distribution
                logging.info(f"Plotting output distribution histogram...")
                pt_out_norm = (pt_out - pt_out_mean) / pt_out_std
                plot_pt_distribution(
                    data=pt_out_norm.numpy(),
                    output_path=output_path,
                    power_transform_id=power_transform_id,
                    lmbda=lmbda,
                    kl_div=kl_div,
                    attribute=attribute,
                    histogram_bins=args.histogram_bins
                )
                # Save output distribution
                numpy_pt_out_filename = (f'pt_out_norm_{power_transform_id.lower()}_{attribute}'
                                         f'_lambda_{lmbda:.2f}_kld_{kl_div:.2f}.npy')
                logging.info(f"Saving output distribution to numpy file {numpy_pt_out_filename}....")
                np.save(numpy_output_path / numpy_pt_out_filename, pt_out_norm)
            # Plot KLD as a function of lambda
            logging.info(f"Plotting KLD as a function of lambda...")
            plot_kld_lambda_fn(lambda_range, kl_divs, output_path, power_transform_id, attribute)
            # Save KLD and lambda range
            numpy_klds_filename = (f'klds_{power_transform_id.lower()}_{attribute}_'
                                   f'lmin_{lambda_min}_lmax_{lambda_max}_lstep_{lambda_step}.npy')
            logging.info(f"Saving KLD and lambda range to numpy file {numpy_klds_filename}....")
            np.save(numpy_output_path / numpy_klds_filename, [lambda_range, kl_divs])


def test_original_distributions(args):
    for attribute in args.attributes:
        dataset = utilities.load_dataset(dataset_path=args.dataset_path,
                                         sequence_length=args.sequence_length,
                                         attribute=attribute,
                                         batch_size=args.batch_size,
                                         parse_sequence_feature=False)
        attribute_data = []
        for batch in dataset:
            attribute_data.append(batch.numpy())
        attribute_data = np.concatenate(attribute_data, axis=0)
        output_path = Path(args.histogram_output_path) / "original" / attribute
        plot_original_distributions(attribute_data, output_path, attribute, args.histogram_bins)


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


def plot_pt_distribution(data,
                         output_path: Path,
                         power_transform_id: str,
                         lmbda: float,
                         kl_div: float,
                         attribute: str,
                         histogram_bins: List[int]):
    histograms_output_path = output_path / "histograms"
    histograms_output_path.mkdir(parents=True, exist_ok=True)
    pt_title = power_transform_id
    attr_title = attribute.replace("_", " ").capitalize()
    for bins in histogram_bins:
        filename = (f'{str(histograms_output_path)}/histogram_{power_transform_id.lower()}_{attribute}_'
                    f'lambda_{lmbda:.2f}_kld_{kl_div:.2f}_bins_{bins}.png')
        plt.hist(data, bins=bins, color='blue', alpha=0.7)
        plt.suptitle(f'{pt_title} - {attr_title}')
        plt.title(r'$\lambda$ = ' + f'{lmbda:.2f} - KLD = {kl_div:.2f} - Bins = {bins}')
        plt.grid(linestyle=':')
        plt.savefig(filename, format='png', dpi=300)
        plt.close()


def plot_kld_lambda_fn(lambds, klds, output_path: Path, power_transform_id: str, attribute: str):
    pt_title = power_transform_id
    attr_title = attribute.replace("_", " ").capitalize()
    filename = f'{str(output_path)}/kld_vs_lmbda_plot_{power_transform_id.lower()}_{attribute}.png'
    plt.plot(lambds, klds)
    plt.title(f'{pt_title} - {attr_title}')
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'KLD($\lambda$)')
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
    parser.add_argument('--lambda-min', nargs='+', help='Lower value of lambda for the power transform.',
                        default=-2.0, required=False, type=float)
    parser.add_argument('--lambda-max', nargs='+', help='Maximum value of lambda for the power transform.',
                        default=2.0, required=False, type=float)
    parser.add_argument('--lambda-step', nargs='+', help='Increment step for lambda at each iteration.',
                        default=0.25, required=False, type=float)
    parser.add_argument('--logging-level', help='Set the logging level.', default="INFO", required=False,
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO"])
    os.environ["KERAS_BACKEND"] = "tensorflow"
    vargs = parser.parse_args()
    logging.getLogger().setLevel(vargs.logging_level)
    test_original_distributions(vargs)
    test_power_transforms(vargs)
