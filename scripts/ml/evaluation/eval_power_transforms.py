import argparse
import functools
import logging
import os
from pathlib import Path
from typing import List

import keras
import keras.ops as k_ops
import matplotlib.pyplot as plt
import numpy as np
from resolv_ml.utilities.distributions.power_transforms import BoxCox, YeoJohnson
from resolv_pipelines.data.loaders import TFRecordLoader
from resolv_pipelines.data.representation.mir import PitchSequenceRepresentation


def eval_power_transforms(args) -> List[float]:
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

    lambda_range = k_ops.arange(args.lambda_min, args.lambda_max, args.lambda_step)
    for attribute in args.attributes:
        for power_transform_id in args.power_transform_ids:
            output_path = Path(args.histogram_output_path) / power_transform_id.lower() / attribute
            output_path.mkdir(parents=True, exist_ok=True)
            numpy_output_path = output_path / "numpy"
            numpy_output_path.mkdir(parents=True, exist_ok=True)
            kl_divs = []
            for lmbda in lambda_range:
                logging.info(f"Evaluating {power_transform_id} power transform with lambda {lmbda:.2f} for attribute "
                             f"'{attribute}'...")
                # Create PowerTransform model
                pt_model = PowerTransformModel(power_transform_id, lmbda)
                pt_model.trainable = False
                pt_model.compile()
                # Load dataset and compute PowerTransform
                dataset = _load_dataset(dataset_path=args.dataset_path, attribute=attribute, batch_size=args.batch_size)
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
                _plot_distribution(
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
            _plot_kld_lambda_fn(lambda_range, kl_divs, output_path, power_transform_id, attribute)
            # Save KLD and lambda range
            numpy_klds_filename = (f'klds_{power_transform_id.lower()}_{attribute}_'
                                   f'lmin_{args.lambda_min}_lmax_{args.lambda_max}_lstep_{args.lambda_step}.npy')
            logging.info(f"Saving KLD and lambda range to numpy file {numpy_klds_filename}....")
            np.save(numpy_output_path / numpy_klds_filename, [lambda_range, kl_divs])


def _load_dataset(dataset_path: str, attribute: str, batch_size: int):
    representation = PitchSequenceRepresentation(sequence_length=64)
    tfrecord_loader = TFRecordLoader(
        file_pattern=dataset_path,
        parse_fn=functools.partial(
            representation.parse_example,
            attributes_to_parse=[attribute],
            parse_sequence_feature=False
        ),
        # Since there may be 0 valued attributes, add an epsilon to everything in order to avoid problems with the
        # BoxCox Transform computation
        map_fn=lambda x, _: x[attribute] + keras.backend.epsilon(),
        batch_size=batch_size,
        batch_drop_reminder=True,
        deterministic=True
    )
    return tfrecord_loader.load_dataset()


def _plot_distribution(data,
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


def _plot_kld_lambda_fn(lambds, klds, output_path: Path, power_transform_id: str, attribute: str):
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
    parser.add_argument('--attributes', nargs='+', help='Attributes to evaluate.', required=True)
    parser.add_argument('--histogram-output-path', help='Path where the histograms will be saved.', required=True)
    parser.add_argument('--histogram-bins', help='Number of bins for the histogram.',  nargs='+', default=[60],
                        required=False, type=int)
    parser.add_argument('--batch-size', help='Batch size.', required=False, default=64, type=int,
                        choices=[32, 64, 128, 256, 512])
    parser.add_argument('--power-transform-ids', help="IDs of power transforms to evaluate.", nargs='+',
                        choices=["box-cox", "yeo-johnson"], default=["box-cox", "yeo-johnson"], required=False)
    parser.add_argument('--lambda-min', help='Lower value of lambda for the power transform.',
                        default=-2.0, required=False, type=float)
    parser.add_argument('--lambda-max', help='Maximum value of lambda for the power transform.',
                        default=2.0, required=False, type=float)
    parser.add_argument('--lambda-step', help='Increment step for lambda at each iteration.',
                        default=0.25, required=False, type=float)
    parser.add_argument('--logging-level', help='Set the logging level.', default="INFO", required=False,
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO"])
    os.environ["KERAS_BACKEND"] = "tensorflow"
    logging.getLogger().setLevel(logging.INFO)
    eval_power_transforms(parser.parse_args())
