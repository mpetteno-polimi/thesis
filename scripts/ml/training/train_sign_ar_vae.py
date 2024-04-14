import keras.losses
from resolv_ml.models.dlvm.vae.ar_vae import SignAttributeRegularization

from scripts.ml import utilities


if __name__ == '__main__':
    arg_parser = utilities.get_arg_parser(description="Train AR-VAE model with sign attribute regularization.")
    arg_parser.add_argument('--scale-factor', help='Scale factor for tanh in sign regularization loss.', default=1.0,
                            type=float)
    args = arg_parser.parse_args()

    train_data, val_data, input_shape = utilities.load_datasets(
        train_dataset_path=args.train_dataset_path,
        val_dataset_path=args.val_dataset_path,
        attribute=args.attribute
    )

    vae = utilities.get_hierarchical_model(
        attribute_reg_layer=SignAttributeRegularization(
            loss_fn=keras.losses.mean_absolute_error,
            regularization_dimension=args.reg_dim,
            gamma=args.gamma,
            scale_factor=args.scale_factor
        )
    )
    vae.build(input_shape)

    trainer = utilities.get_trainer(vae)
    history = trainer.train(train_data=train_data, validation_data=val_data)
