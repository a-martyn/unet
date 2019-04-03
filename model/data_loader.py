from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
import numpy as np

"""
Data Loader:
Loads the membrane cell segmentation dataset

Adapted and simplified from: 
https://github.com/zhixuhao/unet/blob/master/data.py
"""

# Data Augmentation config
# ----------------------------------------------------------------------------
input_generator_train = ImageDataGenerator(
    rotation_range=2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    brightness_range=[0.8, 1.2],
    rescale=1./255,           #  rescale pixel vals 0-255 --> 0.0-1.0
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='reflect', #nearest
    data_format='channels_last',
    validation_split=0.0
)
target_generator_train = ImageDataGenerator(
    rotation_range=2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    # No brightness transform on target mask
    rescale=1./255,           #  rescale pixel vals 0-255 --> 0.0-1.0
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='reflect',
    data_format='channels_last',
    validation_split=0.0
)

input_generator_test = ImageDataGenerator(
    rescale=1./255,           #  rescale pixel vals 0-255 --> 0.0-1.0
    fill_mode='reflect',
    data_format='channels_last',
    validation_split=0.0
)

target_generator_test = ImageDataGenerator(
    rescale=1./255,           #  rescale pixel vals 0-255 --> 0.0-1.0
    fill_mode='reflect',
    data_format='channels_last',
    validation_split=0.0
)


# Data Loaders
# ----------------------------------------------------------------------------

def adjust_data(y):
    """
    Normalise pixel values and force target mask values to binary vals
    Adapted from: https://github.com/zhixuhao/unet/blob/master/data.py
    Note: above implementation includes support for multi-class labels which
    is excluded here for simplicity
    """
    # Push mask target values to binary
    # Note: this only makes sense if target is grayscale mask
    y[y > y.max()/2] = 1
    y[y <= y.max()/2] = 0 
    return y 


def loader(directory, input_gen, target_gen, batch_sz=2, img_sz=(256, 256)):

    input_subdir = 'input'
    target_subdir = 'target'

    # Input generator
    x_gen = input_gen.flow_from_directory(
        directory,
        target_size=img_sz,
        color_mode="grayscale",
        classes=[input_subdir],
        class_mode=None,
        batch_size=batch_sz,
        seed=1,
        interpolation='nearest'
    )
    
    # Target generator
    y_gen = input_gen.flow_from_directory(
        directory,
        target_size=img_sz,
        color_mode="grayscale",
        classes=[target_subdir],
        class_mode=None,
        batch_size=batch_sz,
        seed=1,
        interpolation='nearest'
    )

    generator = zip(x_gen, y_gen)
    for (x, y) in generator:
        x, y = x, adjust_data(y)
        yield (x, y)


def show_augmentation(img_filepath, imageDataGenerator, n_rows=1):
    n_cols = 4
    img = load_img(img_filepath)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape) 
    
    fig = plt.figure(figsize=(16, 8))
    i = 1
    for batch in imageDataGenerator.flow(x, batch_size=1, seed=1):
        ax = fig.add_subplot(n_rows, n_cols, i)
        ax.imshow(batch[0])
        ax.axis('off')
        i += 1
        if i > n_rows*n_cols: break
    plt.show();
    return


def show_sample(generator):
    batch = next(generator)
    x = batch[0][0]
    y = batch[1][0]
    
    size = (5, 5)
    plt.figure(figsize=size)
    plt.imshow(x[:, :, 0], cmap='gray')
    plt.show()
    plt.figure(figsize=size)
    plt.imshow(y[:, :, 0], cmap='gray')
    plt.show();
    return