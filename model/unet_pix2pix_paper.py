import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler


def unet_pix2pix_paper(input_size=(256, 256, 1), output_channels=1):
    
    """
    A Keras/Tensorflow implementation of the U-net used in pix2pix as described by 
    Isola et. al in, "Image-to-Image Translation with Conditional Adversarial 
    Networks":
    https://arxiv.org/abs/1611.07004
    https://github.com/phillipi/pix2pix/blob/master/models.lua
    
    This architecture is used as the Generator in the pix2pix GAN. It is similar
    to the original U-Net architecture with some notable modifications:
    - addition of batch normalisation after each convolution
    - only a single convolution in each upsampling/downsampling 'block' as opposed
      to 3 conv layers per block in original
    - An even number of center layers at bottom of unet and more of them
    - Use of LeakyReLU instead of ReLU for encoder layer activations
    - convolutional stride 2, and kernels size 4 used everywhere as instead of
      2/1 stride and kernel size 3 in original
    """
    # ----------------------------------------------------------------
    # SETTINGS
    
    # Convolutional layer
    conv_kwargs = {
        'kernel_size': (4, 4),
        'strides': 2,
        'padding': 'same',
        'kernel_initializer': 'he_normal',
        'data_format': 'channels_last'  # (batch, height, width, channels)
    }

    # strides = 2
    # ks = (4, 4) 
    # pad = 'same' # Todo: check padding, should be 1
    # df = "channels_last"
    
    # Batch Normalisation
    # TODO: Instance Normalisation: batchnorm use test stats at test time
    # TODO: pytorch implementation has learnable params, how to implement in keras?
    # bn_axis = -1    # because data_loader returns channels last
    # bn_mmtm = 0.9   # equivalent to pytorch defaults used by author (0.1 in pytorch -> 0.9 in keras/tf)
    # bn_eps  = 1e-5  # match pytorch defaults
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

    # - Let Ck denote a Convolution-BatchNorm-ReLU layerwith k filter
    # - Convolutions downsample by a factor of 2
    # - All ReLUs are leaky
    # Architecture is then:
    # C64-C128-C256-C512-C512-C512-C512-C512

    # TODO: check padding is correct

    # input
    # (256 x 256 x input_channels)

    # layer 1 - C64
    # Batch-Norm is not applied to the first C64 layer
    inputs = Input(input_size)
    e1 = Conv2D(64, **conv_kwargs)(inputs)
    # (128 x 128 x 64)

    # layer 2 - C128
    e2 = LeakyReLU(alpha=slope)(e1)
    e2 = Conv2D(128, **conv_kwargs)(e2)
    e2 = BatchNormalization(**bn_kwargs)(e2)
    # (64 x 64 x 128)

    # layer 3 - C256
    e3 = LeakyReLU(alpha=slope)(e2)
    e3 = Conv2D(256, **conv_kwargs)(e3)
    e3 = BatchNormalization(**bn_kwargs)(e3)
    # (32 x 32 x 256)

    # layer 4 - C512
    e4 = LeakyReLU(alpha=slope)(e3)
    e4 = Conv2D(512, **conv_kwargs)(e4)
    e4 = BatchNormalization(**bn_kwargs)(e4)
    # (16 x 16 x 512)

    # layer 5 - C512
    e5 = LeakyReLU(alpha=slope)(e4)
    e5 = Conv2D(512, **conv_kwargs)(e5)
    e5 = BatchNormalization(**bn_kwargs)(e5)
    # (8 x 8 x 512)

    # layer 6 - C512
    e6 = LeakyReLU(alpha=slope)(e5)
    e6 = Conv2D(512, **conv_kwargs)(e6)
    e6 = BatchNormalization(**bn_kwargs)(e6)
    # (4 x 4 x 512)

    # layer 7 - C512
    e7 = LeakyReLU(alpha=slope)(e6)
    e7 = Conv2D(512, **conv_kwargs)(e7)
    e7 = BatchNormalization(**bn_kwargs)(e7)
    # (2 x 2 x 512)

    # layer 8 - C512
    e8 = LeakyReLU(alpha=slope)(e7)
    e8 = Conv2D(512, **conv_kwargs)(e8)
    # e8 = BatchNormalization(**bn_kwargs)(e8)  # not implemented by authors
    # (1 x 1 x 512)

    # ----------------------------------------------------------------
    # DECODER

    # - Ck denotes a Convolution-BatchNorm-ReLU layerwith k filters:
    # - CDk denotes a a Convolution-BatchNorm-Dropout-ReLU layer
    #   with a dropout rate of 50%
    # - Convolutions upsample by a factor of 2,
    # - All ReLUs are not leaky
    # Architecture is then:
    # CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128

    # layer 9 - CD512
    d1 = ReLU()(e8)
    d1 = Conv2DTranspose(512, **conv_kwargs)(d1)
    d1 = BatchNormalization(**bn_kwargs)(d1)
    d1 = Dropout(dropout)(d1)  # Note: pytorch pix2pix doesn't implement
    # (2 x 2 x 512)

    # layer 10 - CD1024
    d2 = concatenate([d1, e7], axis=-1)
    d2 = ReLU()(d2)
    d2 = Conv2DTranspose(512, **conv_kwargs)(d2)
    d2 = BatchNormalization(**bn_kwargs)(d2)
    d2 = Dropout(dropout)(d2)
    # (4 x 4 x 512)

    # layer 11 - CD1024
    d3 = concatenate([d2, e6], axis=-1)
    d3 = ReLU()(d3)
    d3 = Conv2DTranspose(512, **conv_kwargs)(d3)
    d3 = BatchNormalization(**bn_kwargs)(d3)
    d3 = Dropout(dropout)(d3)
    # (8 x 8 x 512)

    # layer 12 - C1024
    d4 = concatenate([d3, e5], axis=-1)
    d4 = ReLU()(d4)
    d4 = Conv2DTranspose(512, **conv_kwargs)(d4)
    d4 = BatchNormalization(**bn_kwargs)(d4)
    # (16 x 16 x 512)

    # layer 13 - C1024
    d5 = concatenate([d4, e4], axis=-1)
    d5 = ReLU()(d5)
    # Note: authors implement an incongruous shape drop here:
    # (16 x 16 x 1024) -> (32 x 32 x 256)
    d5 = Conv2DTranspose(256, **conv_kwargs)(d5)
    d5 = BatchNormalization(**bn_kwargs)(d5)
    # (32 x 32 x 256)

    # layer 14 - C512
    d6 = concatenate([d5, e3], axis=-1)
    d6 = ReLU()(d6)
    d6 = Conv2DTranspose(128, **conv_kwargs)(d6)
    d6 = BatchNormalization(**bn_kwargs)(d6)
    # (64 x 64 x 128)

    # layer 15 - C256
    d7 = concatenate([d6, e2], axis=-1)
    d7 = ReLU()(d7)
    d7 = Conv2DTranspose(64, **conv_kwargs)(d7)
    d7 = BatchNormalization(**bn_kwargs)(d7)
    # (128 x 128 x 64)

    # In the last layer in the decoder, a convolution is applied 
    # to map to the number of output channels (3 in general, except 
    # in colorization, where it is 2), followed by a Tanh function. 
    # layer 16 - C128
    d8 = concatenate([d7, e1], axis=-1)
    d8 = ReLU()(d8)  
    d8 = Conv2DTranspose(output_channels, **conv_kwargs)(d8)
    d8 = Activation('tanh')(d8)
    # (256 x 256 x output_channels)

    model = Model(inputs=[inputs], outputs=[d8])
    return model
