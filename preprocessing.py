import os
import numpy as np 
from PIL import Image

"""
## Upscale test targets to 512px and remove interpolation

Because *test* targets provided here:
https://github.com/zhixuhao/unet/tree/master/data/membrane/test
...are 256px with interpolation for some reason, 
and these render with artifacts via Keras dataloader because .convert('L) 
is called somewhere I guess

Note: This is hack because I can't find target image without interpolation. 
Removing infomation will be lost, so we should expect test scores to be lower
than competition results
"""


src_pth = ('./data/membrane/test/target256/')
tgt_pth = ('./data/membrane/test/target/')
filenames = os.listdir(src_pth)

for f in filenames:
    img = Image.open(f'{src_pth}{f}')
    # upscale
    img = img.resize((512, 512), resample=Image.BOX)
    # Remove interpolation, set all pixels to either full black or white
    arr = np.array(img)
    arr[arr < arr.max()/2] = arr.min()
    arr[arr >= arr.max()/2] = arr.max()
    img = Image.fromarray(arr)
    
    img.save(f'{tgt_pth}{f}')