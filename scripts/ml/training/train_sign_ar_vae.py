"""
Usage example:

    python ./scripts/ml/training/train_sign_ar_vae.py \
        --model-config-path=./scripts/ml/training/config/model.json \
        --trainer-config-path=./scripts/ml/training/config/trainer.json \
        --train-dataset-config-path=./scripts/ml/training/config/train_dataset.json \
        --val-dataset-config-path=./scripts/ml/training/config/val_dataset.json \
        --attribute="contour" \
        --reg-dim=0 \
        --gamma=1.0 \
        --scale-factor=1.0 \
        --gpus=0

"""
import logging

import keras
from resolv_ml.models.dlvm.vae.ar_vae import SignAttributeRegularization
from resolv_ml.training.callbacks import LearningRateLoggerCallback

from scripts.ml.training import utilities


if __name__ == '__main__':
    arg_parser = utilities.get_arg_parser(description="Train AR-VAE model with sign attribute regularization.")
    arg_parser.add_argument('--attribute', help='Attribute to regularize.', required=True)
    arg_parser.add_argument('--reg-dim', help='Latent code regularization dimension.', default=0, type=int)
    arg_parser.add_argument('--gamma', help='Gamma factor to scale regularization loss.', default=1.0, type=float)
    arg_parser.add_argument('--scale-factor', help='Scale factor for tanh in sign regularization loss.', default=1.0,
                            type=float)
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
        vae = utilities.get_model(
            model_config_path=args.model_config_path,
            hierarchical_decoder=args.hierarchical_decoder,
            attribute_reg_layer=SignAttributeRegularization(
                loss_fn=keras.losses.MeanAbsoluteError(),
                regularization_dimension=args.reg_dim,
                gamma=args.gamma,
                scale_factor=args.scale_factor
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
