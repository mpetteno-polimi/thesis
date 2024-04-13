import keras.losses
from resolv_ml.models.dlvm.vae.ar_vae import SignAttributeRegularization

from scripts.ml import utilities


if __name__ == '__main__':

    train_data, val_data = utilities.load_datasets(
        train_dataset_path="",  # TODO - script parameter
        val_dataset_path="",  # TODO - script parameter
        attribute="",   # TODO - script parameter
    )

    vae = utilities.get_hierarchical_model(
        attribute_reg_layer=SignAttributeRegularization(
            loss_fn=keras.losses.mean_absolute_error,
            regularization_dimension=0,  # TODO - script parameter
            gamma=1.0,  # TODO - script parameter
            scale_factor=1.0  # TODO - script parameter
        )
    )

    trainer = utilities.get_trainer(vae)
    history = trainer.train(train_data=train_data, validation_data=val_data)
