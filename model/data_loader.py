from tensorflow.keras.preprocessing.image import ImageDataGenerator

"""
Data Loader:
Loads the membrane cell segmentation dataset

Adapted and simplified from: 
https://github.com/zhixuhao/unet/blob/master/data.py
"""

def adjust_data(x, y):
    """
    Normalise pixel values and force target mask values to absolute vals
    Adapted from: https://github.com/zhixuhao/unet/blob/master/data.py
    Note: above implementation includes support for mult-class labels which
    is excluded here for simplicity
    """
    # Int to floats
    x = x / 255
    y = y / 255
    # Push mask target values to absolute
    # Note: this only makes sense if target is grayscale mask
    y[y > y.max()/2] = 1
    y[y <= y.max()/2] = 0 
    return(x, y) 


def generator(directory, batch_sz=2, img_sz=(256, 256), transforms:dict={}):
    
    x_datagen = ImageDataGenerator(**transforms)
    y_datagen = ImageDataGenerator(**transforms)

    input_subdir = 'input'
    target_subdir = 'target'

    # Input generator
    x_gen = x_datagen.flow_from_directory(
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
    y_gen = x_datagen.flow_from_directory(
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
        x, y = adjust_data(x, y)
        yield (x, y)
