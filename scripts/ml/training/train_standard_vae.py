"""
Usage example:

    python ./scripts/ml/training/train_standard_vae.py \
        --model-config-path=./scripts/ml/training/config/model.json \
        --trainer-config-path=./scripts/ml/training/config/trainer.json \
        --train-dataset-config-path=./scripts/ml/training/config/train_dataset.json \
        --val-dataset-config-path=./scripts/ml/training/config/val_dataset.json \
        --gpus=0

"""
import logging

from resolv_ml.training.callbacks import LearningRateLoggerCallback

from scripts.ml.training import utilities


if __name__ == '__main__':
    arg_parser = utilities.get_arg_parser(description="Train standard VAE model without attribute regularization.")
    args = arg_parser.parse_args()

    logging.getLogger().setLevel(args.logging_level)

    strategy = utilities.get_distributed_strategy(args.gpus)
    with strategy.scope():
        train_data, val_data, input_shape = utilities.load_datasets(
            train_dataset_config_path=args.train_dataset_config_path,
            val_dataset_config_path=args.val_dataset_config_path,
            trainer_config_path=args.trainer_config_path
        )
        vae = utilities.get_model(
            model_config_path=args.model_config_path,
            hierarchical_decoder=args.hierarchical_decoder
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
