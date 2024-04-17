"""
Usage example:

    python ./scripts/ml/training/train_default_ar_vae.py \
        --model-config-path=./scripts/ml/training/config/model.json \
        --trainer-config-path=./scripts/ml/training/config/trainer.json \
        --dataset-config-path=./scripts/ml/training/config/dataset.json \
        --train-dataset-path=./4bars_melodies/train/*.tfrecord \
        --val-dataset-path=./4bars_melodies/validation/*.tfrecord \
        --attribute="contour" \
        --reg-dim=0 \
        --gamma=1.0 \
        --gpus=0

"""
import logging

import keras.losses
from resolv_ml.models.dlvm.vae.ar_vae import DefaultAttributeRegularization

from scripts.ml.training import utilities


if __name__ == '__main__':

    arg_parser = utilities.get_arg_parser(description="Train AR-VAE model with default attribute regularization.")
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
        vae = utilities.get_hierarchical_model(
            model_config_path=args.model_config_path,
            attribute_reg_layer=DefaultAttributeRegularization(
                loss_fn=keras.losses.mean_absolute_error,
                batch_normalization=keras.layers.BatchNormalization(),
                regularization_dimension=args.reg_dim,
                gamma=args.gamma
            )
        )
        vae.build(input_shape)
        trainer = utilities.get_trainer(model=vae, trainer_config_path=args.trainer_config_path)
        history = trainer.train(train_data=train_data, validation_data=val_data)
