"""
Usage example:

    python ./scripts/ml/training/train_power_transform_ar_vae.py \
        --model-config-path=./scripts/ml/training/config/model.json \
        --trainer-config-path=./scripts/ml/training/config/trainer.json \
        --dataset-config-path=./scripts/ml/training/config/dataset.json \
        --train-dataset-path=./4bars_melodies/train/*.tfrecord \
        --val-dataset-path=./4bars_melodies/validation/*.tfrecord \
        --attribute="contour" \
        --reg-dim=0 \
        --power-transform="box-cox" \
        --lambda-init=0.0 \
        --gpus=0

"""
import logging

import keras.losses
from resolv_ml.models.dlvm.vae.ar_vae import PowerTransformAttributeRegularization
from resolv_ml.utilities.distributions.power_transforms import BoxCox, YeoJohnson

from scripts.ml.training import utilities

if __name__ == '__main__':

    arg_parser = utilities.get_arg_parser(description="Train AR-VAE model with sign attribute regularization.")
    arg_parser.add_argument('--power-transform', help='Power transform to use for regularization.', required=True,
                            choices=['box-cox', 'yeo-johnson'])
    arg_parser.add_argument('--lambda-init', help='Initial value for the power transform lambda parameter.',
                            default=0.0, type=float)
    args = arg_parser.parse_args()

    logging.getLogger().setLevel(args.logging_level)

    strategy = utilities.get_distributed_strategy(args.gpus)
    with strategy.scope():
        train_data, val_data, input_shape = utilities.load_datasets(
            dataset_config_path=args.dataset_config_path,
            trainer_config_path=args.trainer_config_path,
            train_dataset_path=args.train_dataset_path,
            val_dataset_path=args.val_dataset_path,
            attribute=args.attribute
        )

        if args.power_transform == "box-cox":
            power_transform_layer = BoxCox(
                lambda_init=args.lambda_init,
                batch_norm=keras.layers.BatchNormalization()
            )
        elif args.power_transform == "yeo-johnson":
            power_transform_layer = YeoJohnson(
                lambda_init=args.lambda_init,
                batch_norm=keras.layers.BatchNormalization()
            )
        else:
            raise ValueError("Power transform must be box-cox or yeo-johnson.")

        vae = utilities.get_hierarchical_model(
            model_config_path=args.model_config_path,
            attribute_reg_layer=PowerTransformAttributeRegularization(
                power_transform=power_transform_layer,
                loss_fn=keras.losses.mean_absolute_error,
                regularization_dimension=args.reg_dim,
                gamma=args.gamma
            )
        )
        vae.build(input_shape)

        trainer = utilities.get_trainer(model=vae, trainer_config_path=args.trainer_config_path)
        history = trainer.train(train_data=train_data, validation_data=val_data)
