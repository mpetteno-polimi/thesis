import keras.losses
from resolv_ml.models.dlvm.vae.ar_vae import PowerTransformAttributeRegularization
from resolv_ml.utilities.distributions.power_transforms import BoxCox, YeoJohnson

from scripts.ml import utilities


if __name__ == '__main__':

    train_data, val_data = utilities.load_datasets(
        train_dataset_path="",  # TODO - script parameter
        val_dataset_path="",  # TODO - script parameter
        attribute="",   # TODO - script parameter
    )

    power_transform = "box-cox"  # TODO - script parameter
    if power_transform == "box-cox":
        power_transform_layer = BoxCox(
            lambda_init=0.0,  # TODO - script parameter
            batch_norm=keras.layers.BatchNormalization()
        )
    elif power_transform == "yeo-johnson":
        power_transform_layer = YeoJohnson(
            lambda_init=1.0,  # TODO - script parameter
            batch_norm=keras.layers.BatchNormalization()
        )
    else:
        raise ValueError("Power transform must be box-cox or yeo-johnson.")

    vae = utilities.get_hierarchical_model(
        attribute_reg_layer=PowerTransformAttributeRegularization(
            power_transform=power_transform_layer,
            loss_fn=keras.losses.mean_absolute_error,
            regularization_dimension=0,  # TODO - script parameter
            gamma=1.0,  # TODO - script parameter
        )
    )

    trainer = utilities.get_trainer(vae)
    history = trainer.train(train_data=train_data, validation_data=val_data)
