"""
Usage example:

    python ./scripts/ml/training/train_power_transform_ar_vae.py \
        --model-config-path=./scripts/ml/training/config/model.json \
        --trainer-config-path=./scripts/ml/training/config/trainer.json \
        --train-dataset-config-path=./scripts/ml/training/config/train_dataset.json \
        --val-dataset-config-path=./scripts/ml/training/config/val_dataset.json \
        --attribute="contour" \
        --reg-dim=0 \
        --power-transform="box-cox" \
        --lambda-init=0.0 \
        --gpus=0

"""
import json
import logging

import keras
from resolv_ml.training.callbacks import LearningRateLoggerCallback
from resolv_ml.utilities.distributions.power_transforms import BoxCox, YeoJohnson
from resolv_ml.utilities.regularizers.attribute import DefaultAttributeRegularizer
from resolv_ml.utilities.schedulers import get_scheduler

from scripts.ml.training import utilities


if __name__ == '__main__':
    arg_parser = utilities.get_arg_parser(description="Train AR-VAE model with sign attribute regularization.")
    arg_parser.add_argument('--attribute', help='Attribute to regularize.', required=True)
    arg_parser.add_argument('--reg-dim', help='Latent code regularization dimension.', default=0, type=int)
    arg_parser.add_argument('--power-transform', help='Power transform to use for regularization.', required=True,
                            choices=['box-cox', 'yeo-johnson'])
    arg_parser.add_argument('--lambda-init', help='Initial value for the power transform lambda parameter.',
                            default=0.0, type=float)
    args = arg_parser.parse_args()

    logging.getLogger().setLevel(args.logging_level)

    strategy = utilities.get_distributed_strategy(args.gpus)
    with strategy.scope():
        train_data, val_data, input_shape = utilities.load_datasets(
            train_dataset_config_path=args.train_dataset_config_path,
            val_dataset_config_path=args.val_dataset_config_path,
            trainer_config_path=args.trainer_config_path,
            attribute=args.attribute
        )
        if args.power_transform == "box-cox":
            power_transform_layer = BoxCox(
                lambda_init=args.lambda_init,
                trainable=False,
                batch_norm=keras.layers.BatchNormalization(scale=False, center=False)
            )
        elif args.power_transform == "yeo-johnson":
            power_transform_layer = YeoJohnson(
                lambda_init=args.lambda_init,
                trainable=False,
                batch_norm=keras.layers.BatchNormalization(scale=False, center=False)
            )
        else:
            raise ValueError("Power transform must be box-cox or yeo-johnson.")

        with open(args.model_config_path) as file:
            model_config = json.load(file)
            schedulers_config = model_config["schedulers"]

        with open(args.trainer_config_path) as file:
            fit_config = json.load(file)["fit"]

        vae = utilities.get_model(
            model_config_path=args.model_config_path,
            trainer_config_path=args.trainer_config_path,
            hierarchical_decoder=args.hierarchical_decoder,
            attribute_proc_layer=power_transform_layer,
            attribute_reg_layer=DefaultAttributeRegularizer(
                beta_scheduler=get_scheduler(
                    schedule_type=schedulers_config["attr_reg_gamma"]["type"],
                    schedule_config=schedulers_config["attr_reg_gamma"]["config"]
                ),
                loss_fn=keras.losses.MeanAbsoluteError(),
                regularization_dimension=args.reg_dim,
                name="pt_attr_regularizer"
            )
        )
        vae.build(input_shape)
        trainer = utilities.get_trainer(model=vae, trainer_config_path=args.trainer_config_path)
        history = trainer.train(
            train_data=train_data[0],
            train_data_cardinality=train_data[1],
            validation_data=val_data[0],
            validation_data_cardinality=val_data[1],
            custom_callbacks=[LearningRateLoggerCallback()]
        )
