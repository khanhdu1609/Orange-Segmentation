Before training this segment model by ELUnet, i have tried to train tradictional UNET. Unet model's result was perfect but 
it consume so much time to predict the image. So i search for the internet anh found the paper ELU-Net: An Efficient and Lightweight U-Net for Medical Image Segmentation of the authors Yunjiao Deng; Yulei Hou; Jiangtao Yan; Daxing Zeng. 

You can read the paper from hear: https://ieeexplore.ieee.org/document/9745574.

The Elunet model have about 650000 parameters compare to 31 milions of tradictional Unet . Therefor, it is so light and just take 0.015s to predict on CPU. 
### Image: 
![alt text](<test img/A1_1.jpeg>)

### Mask:
![alt text](<test img/mask.jpeg>)

### Result:
![alt text](<test img/result.jpeg>)