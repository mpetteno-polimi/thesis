import keras.losses
from resolv_ml.models.dlvm.vae.ar_vae import DefaultAttributeRegularization

from scripts.ml import utilities


if __name__ == '__main__':
    arg_parser = utilities.get_arg_parser(description="Train AR-VAE model with default attribute regularization.")
    args = arg_parser.parse_args()

    train_data, val_data, input_shape = utilities.load_datasets(
        train_dataset_path=args.train_dataset_path,
        val_dataset_path=args.val_dataset_path,
        attribute=args.attribute
    )

    vae = utilities.get_hierarchical_model(
        attribute_reg_layer=DefaultAttributeRegularization(
            loss_fn=keras.losses.mean_absolute_error,
            batch_normalization=keras.layers.BatchNormalization(),
            regularization_dimension=args.reg_dim,
            gamma=args.gamma
        )
    )
    vae.build(input_shape)

    trainer = utilities.get_trainer(vae)
    history = trainer.train(train_data=train_data, validation_data=val_data)
