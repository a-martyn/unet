import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler


def encoder_block(x, filters, kernel_size, downsample=False):
    conv_kwargs = dict(
        activation='relu',
        padding='same',
        kernel_initializer='he_normal',
        data_format='channels_last'  # (batch, height, width, channels)
    )
    
    # Downsample input to halve Height and Width dimensions
    if downsample:
        x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Convolve
    x = Conv2D(filters, kernel_size, **conv_kwargs)(x)
    x = Conv2D(filters, kernel_size, **conv_kwargs)(x)
    return x


def decoder_block(inputs, filters, kernel_size):
    x, shortcut = inputs
    
    conv_kwargs = dict(
        activation='relu',
        padding='same',
        kernel_initializer='he_normal',
        data_format='channels_last'  # (batch, height, width, channels)
    )
    
    # Upsample input to double Height and Width dimensions
    # Concatenate u-net shortcut to input
    up = Conv2D(filters, 2, **conv_kwargs)(UpSampling2D(size=(2, 2))(x))
    x = concatenate([shortcut, up], axis=3)
    
    # Convolve
    x = Conv2D(filters, kernel_size, **conv_kwargs)(x)
    x = Conv2D(filters, kernel_size, **conv_kwargs)(x)
    return x


def unet_baseline(input_size=(256, 256, 1)):
    """U-net implementation adapted from: https://github.com/zhixuhao/unet"""

    # The U
    inputs = Input(input_size)
    e1 = encoder_block(inputs, 64, 3, downsample=False)
    e2 = encoder_block(e1, 128, 3, downsample=True)
    e3 = encoder_block(e2, 256, 3, downsample=True)
    e4 = encoder_block(e3, 512, 3, downsample=True)
    e4 = Dropout(0.5)(e4)
    
    e5 = encoder_block(e4, 1024, 3, downsample=True)
    e5 = Dropout(0.5)(e5)

    d6 = decoder_block([e5, e4], 512, 3)
    d7 = decoder_block([d6, e3], 256, 3)
    d8 = decoder_block([d7, e2], 128, 3)
    d9 = decoder_block([d8, e1], 64,  3)

    # Ouput
    op = Conv2D(2, 3, padding='same', kernel_initializer='he_normal')(d9)
    op = ReLU()(op)
    op = Conv2D(1, 1)(op)
    op = Activation('sigmoid')(op)

    # Build
    model = Model(inputs=[inputs], outputs=[op])
    return model




