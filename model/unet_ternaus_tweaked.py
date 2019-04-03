import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.engine.network import Network
from tensorflow.keras import backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.applications.vgg16 import VGG16

"""
A tweak of TernausNet16 to add dropout and batch normalisation: 
https://arxiv.org/abs/1801.05746
https://github.com/ternaus/TernausNet
"""


def decoder_block_ternausV2(inputs, mid_channels, out_channels, 
                            batch_norm=True):
    """
    Decoder block as proposed for TernausNet16: 
    https://arxiv.org/abs/1801.05746
    See DecoderBlockV2 here:
    https://github.com/ternaus/TernausNet/blob/master/unet_models.py

    - Concatenate u-net shortcut to input pre-upsample
    - Bilinear upsample input to double Height and Width dimensions
    - Note: The original ternausNet implementation includes option for 
      deconvolution instead of bilinear upsampling. Omitted here because I 
      couldn't find a performance comparison
    """
    
    conv_kwargs = dict(
        activation='relu',
        padding='same',
        kernel_initializer='he_normal',
        data_format='channels_last'  # (batch, height, width, channels)
    )
   
    bn_kwargs = {
        'axis': -1,       # because data_loader returns channels last
        'momentum': 0.9,  # equivalent to pytorch defaults used by author (0.1 in pytorch -> 0.9 in keras/tf)
        'epsilon': 1e-5   # match pytorch defaults
    }

    x = UpSampling2D(size=(2, 2))(inputs)
    x = Conv2D(mid_channels, 3, **conv_kwargs)(x)
    if batch_norm: x = BatchNormalization(**bn_kwargs)(x)
    x = Conv2D(out_channels, 3, **conv_kwargs)(x)
    if batch_norm: x = BatchNormalization(**bn_kwargs)(x)
    return x


def reset_weights(model):
    """
    Resets model weights by re-initialising them for each layer.
    See discussion: https://github.com/keras-team/keras/issues/341
    """
    session = K.get_session()
    for layer in model.layers: 
        if isinstance(layer, Network):
            reset_weights(layer)
            continue
        # Only reset vgg pretrained conv layers
        if layer.name[:5] == 'block':
            for v in layer.__dict__:
                v_arg = getattr(layer, v)
                if hasattr(v_arg,'initializer'):
                    initializer_method = getattr(v_arg, 'initializer')
                    initializer_method.run(session=session)
                    print('reinitializing layer {}.{}'.format(layer.name, v))


# INTENDED API
# ----------------------------------------------------------------------------

def ternausNet16_tweaked(input_size=(256, 256, 3), output_channels=1, 
                         dropout=True, batch_norm=True, pretrained=True):
    """
    A Keras implementation of TernausNet16: 
    https://arxiv.org/abs/1801.05746
    https://github.com/ternaus/TernausNet
    """
    bn_kwargs = {
        'axis': -1,       # because data_loader returns channels last
        'momentum': 0.9,  # equivalent to pytorch defaults used by author (0.1 in pytorch -> 0.9 in keras/tf)
        'epsilon': 1e-5   # match pytorch defaults
    }
    # input 
    # convert 1 channel grayscale to 3 channels if needed
    inputs = Input(input_size)
    if input_size[-1] < 3:
        x = Conv2D(3, 1)(inputs)                         # add channels
        input_shape = (input_size[0], input_size[0], 3)  # update input size
    else:
        x = inputs
        input_shape = input_size
    
    # Load pretrained VGG, conv layers include relu activation
    encoder = VGG16(include_top=False, weights='imagenet',
                    input_shape=input_shape)
       
    # (None, 256, 256, 3)
    e1 = encoder.get_layer(name='block1_conv1')(x)
    e1 = encoder.get_layer(name='block1_conv2')(e1)
    # (None, 256, 256, 64)
    e2 = MaxPooling2D(pool_size=(2, 2))(e1)
    e2 = encoder.get_layer(name='block2_conv1')(e2)
    if batch_norm: e2 = BatchNormalization(**bn_kwargs)(e2)
    e2 = encoder.get_layer(name='block2_conv2')(e2)
    if batch_norm: e2 = BatchNormalization(**bn_kwargs)(e2)
    # (None, 128, 128, 128)
    e3 = MaxPooling2D(pool_size=(2, 2))(e2)
    e3 = encoder.get_layer(name='block3_conv1')(e3)
    if batch_norm: e3 = BatchNormalization(**bn_kwargs)(e3)
    e3 = encoder.get_layer(name='block3_conv2')(e3)
    if batch_norm: e3 = BatchNormalization(**bn_kwargs)(e3)
    e3 = encoder.get_layer(name='block3_conv3')(e3)
    if batch_norm: e3 = BatchNormalization(**bn_kwargs)(e3)
    # (None, 64, 64, 256)
    e4 = MaxPooling2D(pool_size=(2, 2))(e3)
    e4 = encoder.get_layer(name='block4_conv1')(e4)
    if batch_norm: e4 = BatchNormalization(**bn_kwargs)(e4)
    e4 = encoder.get_layer(name='block4_conv2')(e4)
    if batch_norm: e4 = BatchNormalization(**bn_kwargs)(e4)
    e4 = encoder.get_layer(name='block4_conv3')(e4)
    if batch_norm: e4 = BatchNormalization(**bn_kwargs)(e4)
    # (None, 32, 32, 512)
    e5 = MaxPooling2D(pool_size=(2, 2))(e4)
    e5 = encoder.get_layer(name='block5_conv1')(e5)
    if batch_norm: e5 = BatchNormalization(**bn_kwargs)(e5)
    e5 = encoder.get_layer(name='block5_conv2')(e5)
    if batch_norm: e5 = BatchNormalization(**bn_kwargs)(e5)
    e5 = encoder.get_layer(name='block5_conv3')(e5)
    if batch_norm: e5 = BatchNormalization(**bn_kwargs)(e5)
    if dropout: e5 = Dropout(0.5)(e5)
    # (None, 16, 16, 512)
    center = MaxPooling2D(pool_size=(2, 2))(e5)
    # (None, 8, 8, 512)
    center = decoder_block_ternausV2(center, 512, 256)
    if batch_norm: center = BatchNormalization(**bn_kwargs)(center)
    if dropout: centre = Dropout(0.5)(center)
    # (None, 16, 16, 256)
    d5 = concatenate([e5, center], axis=3)
    d5 = decoder_block_ternausV2(d5, 512, 256, batch_norm=batch_norm)
    # (None, 32, 32, 256)
    d4 = concatenate([e4, d5], axis=3)
    d4 = decoder_block_ternausV2(d4, 512, 128, batch_norm=batch_norm)
    # (None, 64, 64, 128)
    d3 = concatenate([e3, d4], axis=3)
    d3 = decoder_block_ternausV2(d3, 256, 64, batch_norm=batch_norm)
    # (None, 128, 128, 64)
    d2 = concatenate([e2, d3], axis=3)
    d2 = decoder_block_ternausV2(d2, 128, 64, batch_norm=batch_norm)
    # (None, 256, 256, 64)
    # Note: no decoder block used at end
    d1 = concatenate([e1, d2], axis=3)
    d1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(d1)
    d1 = ReLU()(d1)
    # (None, 256, 256, 32)

    # Output
    if output_channels > 1:
        # untested
        op = tf.nn.log_softmax_v2(d1, axis=3)
    else:
        op = Conv2D(output_channels, 1)(d1)
        op = Activation('sigmoid')(op)  # note: ternaus excludes

    # Build
    model = Model(inputs=[inputs], outputs=[op])
    
    # Forget pretrained weights
    if not pretrained:
        reset_weights(model)
    
    return model




