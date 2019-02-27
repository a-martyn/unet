
```
conda env create -f environment.yml
conda activate unet
```

## Questions?

How might "elastic deformation" data augmentation be reproduced?

Does upsampling with transposed-convolution layers improve performance over simple nearest neighbour upsampling?
- unet_baseline(transpose=True)
- unet_baseline(transpose=False)

Does using pretrained VGG network as encoder improve performance?
- unet_ternaus

Does ternausNet benefit from addition of dropout per orginal U-Net?
Does batch normalisation help?
- ternaus net mods

Does ternausNet benefit from addition of LeakyReLU in encoder?


## Applications

