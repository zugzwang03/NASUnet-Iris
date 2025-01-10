import keras_tuner as kt
import dataloaders.__init__ as dataloader
import unet

train = dataloader.train

def build_model(hp):
    return unet.build_unet_model(hp)

tuner = kt.Hyperband(
    build_model,
    objective="val_accuracy",
    max_epochs=10,
    hyperband_iterations=1,
    directory="my_dir",
    project_name="unet_nas",
)

tuner.search(
    train[0],
    train[1],
    epochs=20,
    batch_size=32,
    validation_split=0.2,  # Or use a separate validation set
)
