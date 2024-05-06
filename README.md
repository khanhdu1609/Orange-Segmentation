Before training this segment model by ELUnet, i have tried to train tradictional UNET. Unet model's result was perfect but 
it consume so much time to predict the image. So i search for the internet anh found the paper ELU-Net: An Efficient and Lightweight U-Net for Medical Image Segmentation of the authors Yunjiao Deng; Yulei Hou; Jiangtao Yan; Daxing Zeng. 

You can read the paper from hear: https://ieeexplore.ieee.org/document/9745574.

The Elunet model have about 650000 parameters compare to 31 milions of tradictional Unet . Therefor, it is so light and just take 0.015s to predict on CPU. 

Origin Image

![A1_1](https://github.com/khanhdu1609/Orange-Segmentation/assets/141617409/597a1520-ffd3-469c-b7db-2a0c797e55c4)

Mask image

![mask](https://github.com/khanhdu1609/Orange-Segmentation/assets/141617409/3e7bba7f-c143-4ddf-9370-d4f4a4915ca0)

Result 

![result](https://github.com/khanhdu1609/Orange-Segmentation/assets/141617409/b69842b1-9ae4-4101-a084-3d76c853c1bd)
