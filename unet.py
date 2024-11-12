import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt

class FFTLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        # Step 1: Apply 2D FFT
        fft = tf.signal.fft2d(tf.cast(inputs, tf.complex64))
        magnitude = tf.abs(fft)

        # Step 2: Apply high-pass and low-pass filtering
        highPass = tf.cast((magnitude > 0.30), tf.complex64)
        lowPass = tf.cast((magnitude < 0.30), tf.complex64)

        high_pass_ifft = tf.math.real(tf.signal.ifft2d(highPass))
        low_pass_ifft = tf.math.real(tf.signal.ifft2d(lowPass))
                
        # Concatenate along the last axis
        concatenated = tf.concat([inputs, high_pass_ifft, low_pass_ifft], axis=-1)  # Shape: (height, width, 3)       
        
        concatenatedShape = int(concatenated.shape[3] / 3)
        result = concatenated[:, :, :, :concatenatedShape]
        
        return result

def build_unet_model(hp):
    inputs = Input(shape=(64, 64, 1))  # Adjust input shape if needed
    
    fft = FFTLayer()(inputs)

    # Encoder
    c1 = Conv2D(hp.Int('filters1', min_value=32, max_value=64, step=32), (3, 3), activation='relu', padding='same')(fft)
    c1 = Conv2D(hp.Int('filters1', min_value=32, max_value=64, step=32), (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    p1 = FFTLayer()(p1)
    c2 = Conv2D(hp.Int('filters2', min_value=64, max_value=128, step=64), (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(hp.Int('filters2', min_value=64, max_value=128, step=64), (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    p2 = FFTLayer()(p2)
    c3 = Conv2D(hp.Int('filters3', min_value=128, max_value=256, step=128), (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(hp.Int('filters3', min_value=128, max_value=256, step=128), (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    p3 = FFTLayer()(p3)
    c4 = Conv2D(hp.Int('filters4', min_value=256, max_value=512, step=256), (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(hp.Int('filters4', min_value=256, max_value=512, step=256), (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    
    p4 = FFTLayer()(p4)
    # Bottleneck
    c5 = Conv2D(hp.Int('filters5', min_value=512, max_value=1024, step=512), (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(hp.Int('filters5', min_value=512, max_value=1024, step=512), (3, 3), activation='relu', padding='same')(c5)
    
    c5 = FFTLayer()(c5)
    # Decoder
    d4 = Conv2DTranspose(hp.Int('filters4', min_value=256, max_value=512, step=256), (2, 2), strides=(2, 2), padding='same')(c5)
    d4 = concatenate([d4, c4])
    c6 = Conv2D(hp.Int('filters4', min_value=256, max_value=512, step=256), (3, 3), activation='relu', padding='same')(d4)
    c6 = Conv2D(hp.Int('filters4', min_value=256, max_value=512, step=256), (3, 3), activation='relu', padding='same')(c6)
    
    c6 = FFTLayer()(c6)
    d3 = Conv2DTranspose(hp.Int('filters3', min_value=128, max_value=256, step=128), (2, 2), strides=(2, 2), padding='same')(c6)
    d3 = concatenate([d3, c3])
    c7 = Conv2D(hp.Int('filters3', min_value=128, max_value=256, step=128), (3, 3), activation='relu', padding='same')(d3)
    c7 = Conv2D(hp.Int('filters3', min_value=128, max_value=256, step=128), (3, 3), activation='relu', padding='same')(c7)
    
    c7 = FFTLayer()(c7)
    d2 = Conv2DTranspose(hp.Int('filters2', min_value=64, max_value=128, step=64), (2, 2), strides=(2, 2), padding='same')(c7)
    d2 = concatenate([d2, c2])
    c8 = Conv2D(hp.Int('filters2', min_value=64, max_value=128, step=64), (3, 3), activation='relu', padding='same')(d2)
    c8 = Conv2D(hp.Int('filters2', min_value=64, max_value=128, step=64), (3, 3), activation='relu', padding='same')(c8)
    
    c8 = FFTLayer()(c8)
    d1 = Conv2DTranspose(hp.Int('filters1', min_value=32, max_value=64, step=32), (2, 2), strides=(2, 2), padding='same')(c8)
    d1 = concatenate([d1, c1])
    c9 = Conv2D(hp.Int('filters1', min_value=32, max_value=64, step=32), (3, 3), activation='relu', padding='same')(d1)
    c9 = Conv2D(hp.Int('filters1', min_value=32, max_value=64, step=32), (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model