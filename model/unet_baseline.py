import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

"""
A Keras/Tensorflow implementation of the original U-Net architecture
described by Olaf Ronneberger et. al in "U-Net: Convolutional Networks for
Biomedical Image Segmentation":
paper: https://arxiv.org/abs/1505.04597
code : https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
       (see phseg_v3-train.prototxt for caffe layer descriptions)

What's not implemented?
- The authors eschew padding and as result propose cropping in the upsampling
  layers. Here instead we use padding to avoid the need for cropping.
"""



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


def decoder_block(inputs, filters, kernel_size, transpose=True):
    x, shortcut = inputs
    
    conv_kwargs = dict(
        activation='relu',
        padding='same',
        kernel_initializer='he_normal',
        data_format='channels_last'  # (batch, height, width, channels)
    )
    
    # Upsample input to double Height and Width dimensions
    if transpose:
        # Transposed convolution a.k.a fractionally-strided convolution 
        # or deconvolution although use of the latter term is confused.
        # Excellent explanation: https://github.com/vdumoulin/conv_arithmetic
        up = Conv2DTranspose(filters, 2, strides=2, **conv_kwargs)(x)
    else:
        # Upsampling by simply repeating rows and columns then convolve
        up = UpSampling2D(size=(2, 2))(x)
        up = Conv2D(filters, 2, **conv_kwargs)(up)
    
    # Concatenate u-net shortcut to input
    x = concatenate([shortcut, up], axis=3)
    
    # Convolve
    x = Conv2D(filters, kernel_size, **conv_kwargs)(x)
    x = Conv2D(filters, kernel_size, **conv_kwargs)(x)
    return x


# INTENDED API
# ----------------------------------------------------------------------------

def unet(input_size=(256, 256, 1), output_channels=1, transpose=True):
    """
    U-net implementation adapted translated from authors original
    source code available here: 
    https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
    """

    # The U
    inputs = Input(input_size)
    e1 = encoder_block(inputs, 64, 3, downsample=False)
    e2 = encoder_block(e1, 128, 3, downsample=True)
    e3 = encoder_block(e2, 256, 3, downsample=True)
    e4 = encoder_block(e3, 512, 3, downsample=True)
    e4 = Dropout(0.5)(e4)
    
    e5 = encoder_block(e4, 1024, 3, downsample=True)
    e5 = Dropout(0.5)(e5)

    d6 = decoder_block([e5, e4], 512, 3, transpose=transpose)
    d7 = decoder_block([d6, e3], 256, 3, transpose=transpose)
    d8 = decoder_block([d7, e2], 128, 3, transpose=transpose)
    d9 = decoder_block([d8, e1], 64,  3, transpose=transpose)

    # Ouput
    op = Conv2D(2, 3, padding='same', kernel_initializer='he_normal')(d9)
    op = ReLU()(op)
    op = Conv2D(output_channels, 1)(op)
    op = Activation('sigmoid')(op)

    # Build
    model = Model(inputs=[inputs], outputs=[op])
    return model




