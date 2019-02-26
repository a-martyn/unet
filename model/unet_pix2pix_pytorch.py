import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler


# INTENDED API
# ------------------------------------------------------------------------------

def unet_pix2pix_pytorch(input_size=(256,256,1), output_channels=1):
    """
    A Keras/Tensorflow implementation of the U-net used in the latest pix2pix 
    PyTorch official implementation:
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    
    This architecture is used as the Generator in the pix2pix GAN. It is similar
    to the original U-Net architecture with some notable modifications:
    - addition of batch normalisation after each convolution
    - An even number of center layers at bottom of unet and more of them
    - Use of LeakyReLU instead of ReLU for encoder layer activations
    - convolutional stride 2, and kernels size 4 used everywhere as instead of
      2/1 stride and kernel size 3 in original
    """
    # Convolutional layer
    conv_kwargs = dict(
        padding='same',
        kernel_initializer='he_normal',
        data_format='channels_last'  # (batch, height, width, channels)
    )

    bn_kwargs = {
        'axis': -1,       # because data_loader returns channels last
        'momentum': 0.9,  # equivalent to pytorch defaults used by author (0.1 in pytorch -> 0.9 in keras/tf)
        'epsilon': 1e-5   # match pytorch defaults
    }

    # ReLU
    slope = 0.2
    # Dropout
    dropout = 0.5

    # ----------------------------------------------------------------
    # ENCODER

    # block 1
    inputs = Input(input_size)
    # (256 x 256 x 1)
    e1 = Conv2D(64, 3, strides=2, **conv_kwargs)(inputs)
    e1 = LeakyReLU(alpha=slope)(e1)
    e1 = Conv2D(64, 3, strides=1, **conv_kwargs)(e1)
    # (128 x 128 x 64)

    # block 2 
    e2 = LeakyReLU(alpha=slope)(e1)
    e2 = Conv2D(128, 3, strides=2, **conv_kwargs)(e2)
    e2 = BatchNormalization(**bn_kwargs)(e2)
    e2 = LeakyReLU(alpha=slope)(e2)
    e2 = Conv2D(128, 3, strides=1, **conv_kwargs)(e2)
    e2 = BatchNormalization(**bn_kwargs)(e2)
    # (64 x 64 x 128)

    # block 3 
    e3 = LeakyReLU(alpha=slope)(e2)
    e3 = Conv2D(256, 3, strides=2, **conv_kwargs)(e3)
    e3 = BatchNormalization(**bn_kwargs)(e3)
    e3 = LeakyReLU(alpha=slope)(e3)
    e3 = Conv2D(256, 3, strides=1, **conv_kwargs)(e3)
    e3 = BatchNormalization(**bn_kwargs)(e3)
    # (32 x 32 x 256)

    # block 4
    e4 = LeakyReLU(alpha=slope)(e3)
    e4 = Conv2D(512, 3, strides=2, **conv_kwargs)(e4)
    e4 = BatchNormalization(**bn_kwargs)(e4)
    e4 = LeakyReLU(alpha=slope)(e4)
    e4 = Conv2D(512, 3, strides=1, **conv_kwargs)(e4)
    e4 = BatchNormalization(**bn_kwargs)(e4)
    e4 = Dropout(dropout)(e4)
    # (16 x 16 x 512)

    # block 5a
    e5 = LeakyReLU(alpha=slope)(e4)
    e5 = Conv2D(1024, 3, strides=2, **conv_kwargs)(e5)
    e5 = BatchNormalization(**bn_kwargs)(e5)
    e5 = LeakyReLU(alpha=slope)(e5)
    e5 = Conv2D(1024, 3, strides=1, **conv_kwargs)(e5)
    e5 = BatchNormalization(**bn_kwargs)(e5)
    # (8 x 8 x 1024)


    # ----------------------------------------------------------------
    # DECODER

    # block 6
    d1 = ReLU()(e5)
    d1 = Conv2DTranspose(512, 2, strides=2, **conv_kwargs)(d1)
    d1 = BatchNormalization(**bn_kwargs)(d1)
    d1 = Dropout(dropout)(d1) 
    # (16 x 16 x 512)

    # block 7
    d2 = concatenate([d1, e4], axis=3)
    d2 = ReLU()(d2)
    d2 = Conv2DTranspose(256, 2, strides=2, **conv_kwargs)(d2)
    d2 = BatchNormalization(**bn_kwargs)(d2)
    d2 = ReLU()(d2)
    d2 = Conv2D(256, 3, strides=1, **conv_kwargs)(d2)
    d2 = BatchNormalization(**bn_kwargs)(d2)
    d2 = ReLU()(d2)
    d2 = Conv2D(256, 3, strides=1, **conv_kwargs)(d2)
    d2 = BatchNormalization(**bn_kwargs)(d2)
    # (32 x 32 x 256)

    # block 8
    d3 = concatenate([d2, e3], axis=3)
    d3 = ReLU()(d3)
    d3 = Conv2DTranspose(128, 2, strides=2, **conv_kwargs)(d3)
    d3 = BatchNormalization(**bn_kwargs)(d3)
    d3 = ReLU()(d3)
    d3 = Conv2D(128, 3, strides=1, **conv_kwargs)(d3)
    d3 = BatchNormalization(**bn_kwargs)(d3)
    d3 = ReLU()(d3)
    d3 = Conv2D(128, 3, strides=1, **conv_kwargs)(d3)
    d3 = BatchNormalization(**bn_kwargs)(d3)
    # (64 x 64 x 128)

    # block 9
    d4 = concatenate([d3, e2], axis=3)
    d4 = ReLU()(d4)
    d4 = Conv2DTranspose(64, 2, strides=2, **conv_kwargs)(d4)
    d4 = BatchNormalization(**bn_kwargs)(d4)
    d4 = ReLU()(d4)
    d4 = Conv2D(64, 3, strides=1, **conv_kwargs)(d4)
    d4 = BatchNormalization(**bn_kwargs)(d4)
    d4 = ReLU()(d4)
    d4 = Conv2D(64, 3, strides=1, **conv_kwargs)(d4)
    d4 = BatchNormalization(**bn_kwargs)(d4)
    # (128 x 128 x 64)

    # block 10
    d5 = concatenate([d4, e1], axis=3)
    d5 = ReLU()(d5)  
    d5 = Conv2DTranspose(32, 2, strides=2, **conv_kwargs)(d5)
    d5 = ReLU()(d5)
    d5 = Conv2D(2, 3, strides=1, **conv_kwargs)(d5)
    d5 = ReLU()(d5)
    d5 = Conv2D(1, 1, strides=1, **conv_kwargs)(d5)
    d5 = Activation('tanh')(d5)
    # (256 x 256 x output_channels)

    model = Model(inputs=[inputs], outputs=[d5])
    return model