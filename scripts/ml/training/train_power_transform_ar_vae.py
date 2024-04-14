import keras.losses
from resolv_ml.models.dlvm.vae.ar_vae import PowerTransformAttributeRegularization
from resolv_ml.utilities.distributions.power_transforms import BoxCox, YeoJohnson

from scripts.ml import utilities


if __name__ == '__main__':

    arg_parser = utilities.get_arg_parser(description="Train AR-VAE model with sign attribute regularization.")
    arg_parser.add_argument('--power-transform', help='Power transform to use for regularization.', required=True)
    arg_parser.add_argument('--lambda-init', help='Initial value for the power transform lambda parameter.',
                            default=0.0, type=float)
    args = arg_parser.parse_args()

    train_data, val_data, input_shape = utilities.load_datasets(
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
        attribute_reg_layer=PowerTransformAttributeRegularization(
            power_transform=power_transform_layer,
            loss_fn=keras.losses.mean_absolute_error,
            regularization_dimension=args.reg_dim,
            gamma=args.gamma
        )
    )
    vae.build(input_shape)

    trainer = utilities.get_trainer(vae)
    history = trainer.train(train_data=train_data, validation_data=val_data)
