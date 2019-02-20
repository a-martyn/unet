import numpy as np
# from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
# from tensorflow.keras.layers import UpSampling2D, Dropout, LeakyReLU, ReLU
# from tensorflow.keras.layers import Concatenate, Activation
# from tensorflow.keras.models import Model

import os
import skimage.io as io
import skimage.transform as trans
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from tensorflow.keras import backend as keras


def unet2(pretrained_weights=None, input_size=(256,256,1)):
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


    # NOTES
    # ----------
    # author uses ReLU not LeakyRelu
    # author doesn't use batch norm
    # author uses maxpooling not stride 2

    # ----------------------------------------------------------------
    # ENCODER

    # layer 1
    inputs = Input(input_size)
    e1 = Conv2D(64, 3, strides=1, **conv_kwargs)(inputs)
    e1 = LeakyReLU(alpha=slope)
    e1 = Conv2D(64, 3, strides=1, **conv_kwargs)(e1)
    # (128 x 128 x 64)

    # layer 2 
    e2 = LeakyReLU(alpha=slope)(e1)
    e2 = Conv2D(128, 3, strides=2, **conv_kwargs)(e2)
    e2 = BatchNormalization(**bn_kwargs)(e2)
    e2 = LeakyReLU(alpha=slope)(e2)
    e2 = Conv2D(128, 3, strides=1, **conv_kwargs)(e2)
    e2 = BatchNormalization(**bn_kwargs)(e2)
    # (64 x 64 x 128)

    # layer 3 
    e3 = LeakyReLU(alpha=slope)(e2)
    e3 = Conv2D(256, 3, strides=2, **conv_kwargs)(e3)
    e3 = BatchNormalization(**bn_kwargs)(e3)
    e3 = LeakyReLU(alpha=slope)(e3)
    e3 = Conv2D(256, 3, strides=1, **conv_kwargs)(e3)
    e3 = BatchNormalization(**bn_kwargs)(e3)
    # (32 x 32 x 256)

    # layer 4
    e4 = LeakyReLU(alpha=slope)(e3)
    e4 = Conv2D(512, 3, strides=2, **conv_kwargs)(e4)
    e4 = BatchNormalization(**bn_kwargs)(e4)
    e4 = LeakyReLU(alpha=slope)(e4)
    e4 = Conv2D(512, 3, strides=1, **conv_kwargs)(e4)
    e4 = BatchNormalization(**bn_kwargs)(e4)
    e4 = Dropout(p=dropout)(e4) 
    # (16 x 16 x 512)

    # layer 5a
    e5 = LeakyReLU(alpha=slope)(e4)
    e5 = Conv2D(1024, 3, strides=2, **conv_kwargs)(e5)
    e5 = BatchNormalization(**bn_kwargs)(e5)
    e5 = LeakyReLU(alpha=slope)(e5)
    e5 = Conv2D(1024, 3, strides=1, **conv_kwargs)(e5)
    e5 = BatchNormalization(**bn_kwargs)(e5)
    # (8 x 8 x 1024)


    # ----------------------------------------------------------------
    # DECODER

    # layer 6
    d1 = ReLU()(e5)
    d1 = Conv2DTranspose(512, 2, strides=2, **conv_kwargs)(d1)
    d1 = BatchNormalization(**bn_kwargs)(d1)
    d1 = Dropout(p=dropout)(d1) 
    # (16 x 16 x 512)

    # layer 7
    d2 = Concatenate([d1, e4], axis=-1)
    d2 = ReLU()(d2)
    d2 = Conv2DTranspose(512, 2, strides=2, **conv_kwargs)(d2)
    d2 = BatchNormalization(**bn_kwargs)(d2)
    d2 = ReLU()(d2)
    d2 = Conv2D(512, 3, strides=1, **conv_kwargs)(d2)
    d2 = BatchNormalization(**bn_kwargs)(d2)
    d2 = ReLU()(d2)
    d2 = Conv2D(512, 3, strides=1, **conv_kwargs)(d2)
    d2 = BatchNormalization(**bn_kwargs)(d2)
    # (32 x 32 x 512)

    # layer 8
    d3 = Concatenate([d2, e3], axis=-1)
    d3 = ReLU()(d3)
    d3 = Conv2DTranspose(256, 2, strides=2, **conv_kwargs)(d3)
    d3 = BatchNormalization(**bn_kwargs)(d3)
    d3 = ReLU()(d3)
    d3 = Conv2D(256, 3, strides=1, **conv_kwargs)(d3)
    d3 = BatchNormalization(**bn_kwargs)(d3)
    d3 = ReLU()(d3)
    d3 = Conv2D(256, 3, strides=1, **conv_kwargs)(d3)
    d3 = BatchNormalization(**bn_kwargs)(d3)
    # (64 x 64 x 256)

    # layer 9
    d4 = Concatenate([d3, e2], axis=-1)
    d4 = ReLU()(d4)
    d4 = Conv2DTranspose(512, 2, strides=2, **conv_kwargs)(d4)
    d4 = BatchNormalization(**bn_kwargs)(d4)
    d4 = ReLU()(d4)
    d4 = Conv2D(512, 3, strides=1, **conv_kwargs)(d4)
    d4 = BatchNormalization(**bn_kwargs)(d4)
    d4 = ReLU()(d4)
    d4 = Conv2D(512, 3, strides=1, **conv_kwargs)(d4)
    d4 = BatchNormalization(**bn_kwargs)(d4)
    # (128 x 128 x 128)

    # layer 10
    d5 = Concatenate([d4, e1], axis=-1)
    d5 = ReLU()(d5)  
    d5 = Conv2DTranspose(64, 2, strides=2, **conv_kwargs)(d5)
    d5 = ReLU()(d5)
    d5 = Conv2D(2, 3, strides=1, **conv_kwargs)(d5)
    d5 = ReLU()(d5)
    d5 = Conv2D(1, 3, strides=1, **conv_kwargs)(d5)
    d5 = Activation('tanh')(d5)
    # (256 x 256 x output_channels)

    model = Model(inputs=[e1], outputs=[d5], name='unet')
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def unet(pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model



def unet_pix2pix(input_shape, output_channels):
    
    """
    - All convolutions are 4×4 spatialfilters applied with stride 2.
    
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
    e1 = Conv2D(64, input_shape=input_shape, **conv_kwargs)
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
    d1 = Dropout(p=dropout)(d1)  # Note: pytorch pix2pix doesn't implement
    # (2 x 2 x 512)

    # layer 10 - CD1024
    d2 = Concatenate([d1, e7], axis=-1)
    d2 = ReLU()(d2)
    d2 = Conv2DTranspose(512, **conv_kwargs)(d2)
    d2 = BatchNormalization(**bn_kwargs)(d2)
    d2 = Dropout(p=dropout)(d2)
    # (4 x 4 x 512)

    # layer 11 - CD1024
    d3 = Concatenate([d2, e6], axis=-1)
    d3 = ReLU()(d3)
    d3 = Conv2DTranspose(512, **conv_kwargs)(d3)
    d3 = BatchNormalization(**bn_kwargs)(d3)
    d3 = Dropout(p=dropout)(d3)
    # (8 x 8 x 512)

    # layer 12 - C1024
    d4 = Concatenate([d3, e5], axis=-1)
    d4 = ReLU()(d4)
    d4 = Conv2DTranspose(512, **conv_kwargs)(d4)
    d4 = BatchNormalization(**bn_kwargs)(d4)
    # (16 x 16 x 512)

    # layer 13 - C1024
    d5 = Concatenate([d4, e4], axis=-1)
    d5 = ReLU()(d5)
    # Note: authors implement an incongruous shape drop here:
    # (16 x 16 x 1024) -> (32 x 32 x 256)
    d5 = Conv2DTranspose(256, **conv_kwargs)(d5)
    d5 = BatchNormalization(**bn_kwargs)(d5)
    # (32 x 32 x 256)

    # layer 14 - C512
    d6 = Concatenate([d5, e3], axis=-1)
    d6 = ReLU()(d6)
    d6 = Conv2DTranspose(128, **conv_kwargs)(d6)
    d6 = BatchNormalization(**bn_kwargs)(d6)
    # (64 x 64 x 128)

    # layer 15 - C256
    d7 = Concatenate([d6, e2], axis=-1)
    d7 = ReLU()(d7)
    d7 = Conv2DTranspose(64, **conv_kwargs)(d7)
    d7 = BatchNormalization(**bn_kwargs)(d7)
    # (128 x 128 x 64)

    # In the last layer in the decoder, a convolution is applied 
    # to map to the number of output channels (3 in general, except 
    # in colorization, where it is 2), followed by a Tanh function. 
    # layer 16 - C128
    d8 = Concatenate([d7, e1], axis=-1)
    d8 = ReLU()(d8)  
    d8 = Conv2DTranspose(output_channels, **conv_kwargs)(d8)
    d8 = Activation('tanh')(d8)
    # (256 x 256 x output_channels)

    return Model(inputs=[e1], outputs=[d8], name='unet')

