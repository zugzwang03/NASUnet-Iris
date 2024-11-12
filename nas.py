import keras_tuner as kt
import dataLoader 
import unet

train = dataLoader.train

def build_model(hp):
    return unet.build_unet_model(hp)

tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    hyperband_iterations=5,
    directory='my_dir',
    project_name='unet_nas'
)

tuner.search(
    train[0],
    train[1],
    epochs=10,
    batch_size=64,
    validation_split=0.2  # Or use a separate validation set
)