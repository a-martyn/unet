import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler



def normalisation(x, norm_type='batch'):
    """
    Authors use default batch normalisation settings in PyTorch:
    - `nn.BatchNorm2d(eps=1e-05, momentum=0.1, affine=True, 
                    track_running_stats=True)`
    - By default, gamma is initialised from U(0,1) and elements of β are zero
    - No conv bias is required because affine parameters are used
    - useful ref: https://discuss.pytorch.org/t/ \
      convering-a-batch-normalization-layer-from-tf-to-pytorch/20407
    """

    bn_kwargs = dict(
        axis=-1,         # because data_loader returns channels last
        momentum=0.9,    # equivalent to pytorch defaults used by author (0.1 in pytorch -> 0.9 in keras/tf)
        epsilon=1e-5,    # match pytorch defaults
        beta_initializer='zeros',
        gamma_initializer=tf.initializers.random_uniform(0.0, 1.0), # equivalent to pytorch default
        center=True,     # equivalent to affine=True
        scale=True,      # equivalent to affine=True
        trainable=True,
    )
    
    if norm_type == 'batch':
        x = BatchNormalization(**bn_kwargs)(x)
    elif norm_type == 'none':
        pass
    else:
        raise NotImplementedError(f'norm_type: {norm_type}, not found')
    return x
    

def downconv(x, out_channels, activation=True, norm_type='batch', 
             use_bias=True, init='random_normal', padding=(1, 1)):
    
    conv_kwargs = dict(
        use_bias=use_bias,
        padding='valid',
        kernel_initializer=init,  
        bias_initializer=init,              
        data_format='channels_last'
    )
    
    if activation: x = LeakyReLU(alpha=0.2)(x)
    x = ZeroPadding2D(padding=padding)(x)
    x = Conv2D(out_channels, 4, strides=2, **conv_kwargs)(x)
    x = normalisation(x, norm_type=norm_type)
    return x


def upconv(x, out_channels, norm_type='batch', dropout=False, use_bias=True,
           init='random_normal'):
    
    conv_kwargs = dict(
        use_bias=use_bias,
        padding='same', 
        kernel_initializer=init,  
        bias_initializer=init,                  
        data_format='channels_last'
    )
    
    # Concatenate shortcut and input by channel axis
    if isinstance(x, list):
        x = concatenate(x, axis=-1)
    
    # Transpose convolution
    x = ReLU()(x)
    x = Conv2DTranspose(out_channels, 4, strides=2, **conv_kwargs)(x)
    x = normalisation(x, norm_type=norm_type)
    if dropout: x = Dropout(0.5)(x)
    return x
    

# INTENDED API
# ----------------------------------------------------------------------------

def unet_pix2pix(input_size=(256,256,1), output_channels=1, init_gain=0.02):
    """
    A Keras/Tensorflow implementation of the U-net used in the latest pix2pix 
    PyTorch official implementation:
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    
    This architecture is used as the Generator in the pix2pix GAN. 
    It is similar to the original U-Net architecture with some notable 
    modifications:
    - addition of batch normalisation after each convolution
    - Use of LeakyReLU instead of ReLU for encoder layer activations
    - convolutional stride 2, and kernels size 4 used everywhere instead of
      2/1 stride and kernel size 3 in original
    
    """
    
    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=init_gain)
    oc = output_channels
    nt = 'batch'
    use_bias = False  # affine params in batchnorm so no need for bias
    
    # ----------------------------------------------------------------
    # U-net
    
    # outermost                                                                                   # Output shape
    inputs = Input(input_size)                                                                    # (256, 256, input_size[-1])
    e1 = downconv(inputs, 64, activation=False, norm_type='none', use_bias=use_bias, init=init)   # (128, 128, 64)
    e2 = downconv(e1, 128, activation=True, norm_type=nt, use_bias=use_bias, init=init)           # (64, 64, 128)
    e3 = downconv(e2, 256, activation=True, norm_type=nt, use_bias=use_bias, init=init)           # (32, 32, 256)
    e4 = downconv(e3, 512, activation=True, norm_type=nt, use_bias=use_bias, init=init)           # (16, 16, 512)
    e5 = downconv(e4, 512, activation=True, norm_type=nt, use_bias=use_bias, init=init)           # (8, 8, 512)
    e6 = downconv(e5, 512, activation=True, norm_type=nt, use_bias=use_bias, init=init)           # (4, 4, 512)
    e7 = downconv(e6, 512, activation=True, norm_type=nt, use_bias=use_bias, init=init)           # (2, 2, 512)
    
    # innermost
    e8 = downconv(e7, 512, activation=True, norm_type='none', use_bias=use_bias, 
                  init=init, padding=(1, 1))                                                      # (1 x 1 x 512)
    d8 = upconv(e8, 512, norm_type=nt, dropout=False, use_bias=use_bias, init=init)               # (2 x 2 x 512)
    
    d7 = upconv([d8, e7], 512, norm_type=nt, dropout=True, use_bias=use_bias, init=init)          # (4, 4, 512)
    d6 = upconv([d7, e6], 512, norm_type=nt, dropout=True, use_bias=use_bias, init=init)          # (8, 8, 512)
    d5 = upconv([d6, e5], 512, norm_type=nt, dropout=True, use_bias=use_bias, init=init)          # (16, 16, 512)
    d4 = upconv([d5, e4], 256, norm_type=nt, dropout=False, use_bias=use_bias, init=init)         # (32, 32, 256)
    d3 = upconv([d4, e3], 128, norm_type=nt, dropout=False, use_bias=use_bias, init=init)         # (64, 64, 128)
    d2 = upconv([d3, e2],  64, norm_type=nt, dropout=False, use_bias=use_bias, init=init)         # (128, 128, 64)
    d1 = upconv([d2, e1], oc, norm_type='none', dropout=False, use_bias=True, init=init)          # (256, 256, output_channels)
    op = Activation('tanh', name='G_activations')(d1)

    model = Model(inputs=[inputs], outputs=[op])
    return model