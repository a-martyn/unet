import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler


def encoder_block(filters, kernel_size, downsample=False, x):
    conv_kwargs = dict(
        padding='same',
        kernel_initializer='he_normal',
        data_format='channels_last'  # (batch, height, width, channels)
    )

    # Downsample input to halve Height and Width dimensions
    if downsample:
        x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Convolve
    x = Conv2D(filters, kernel_size, strides=1, **conv_kwargs)(x)
    x = ReLU()(x)
    x = Conv2D(filters, kernel_size, strides=1, **conv_kwargs)(x)
    x = ReLU()(x)
    return x


def decoder_block(filters, kernel_size, upsample=False, [x, shortcut]):
    conv_kwargs = dict(
        padding='same',
        kernel_initializer='he_normal',
        data_format='channels_last'  # (batch, height, width, channels)
    )

    # Upsample input to double Height and Width dimensions
    if upsample:
        x = Conv2D(filters, 2, strides=1, **conv_kwargs)(UpSampling2D(size=(2, 2))(x))
        x = ReLU()(x)
    
    # Concatenate u-net shortcut to input
    x = concatenate([x, shortcut], axis=3)
    # Convolve
    x = Conv2D(filters, kernel_size, strides=1, **conv_kwargs)(x)
    x = ReLU()(x)
    x = Conv2D(filters, kernel_size, strides=1, **conv_kwargs)(x)
    x = ReLU()(x)
    return x


def unet_paper(input_size=(256, 256, 1)):

    # The U
    inputs = Input(input_size)
    e1 = encoder_block(64, 3, downsample=False, inputs)
    e2 = encoder_block(128, 3, downsample=True, e1)
    e3 = encoder_block(256, 3, downsample=True, e2)
    e4 = encoder_block(512, 3, downsample=True, e3)
    e4 = Dropout(0.5)(e4)
    
    e5 = encoder_block(1024, 3, downsample=True, e3)
    e5 = Dropout(0.5)(e5)

    d6 = decoder_block(512, 3, upsample=True, [e5, e4])
    d7 = decoder_block(256, 3, upsample=True, [d6, e3])
    d8 = decoder_block(128, 3, upsample=True, [d7, e2])
    d9 = decoder_block(64,  3, upsample=True, [d8, e1])

    # Ouput
    op = Conv2D(2, 3, padding='same', kernel_initializer='he_normal')(d9)
    op = ReLU()(op)
    op = Conv2D(1, 1)(op)
    op = Activation('sigmoid')(d5)

    # Build
    model = Model(inputs=[inputs], outputs=[d5])
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
    
    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model



