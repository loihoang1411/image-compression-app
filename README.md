# Image_Compression_Py-App
An Image Compression app wrote by Python language, using CustomTkinter

## Table of Contents:
- [Overview](#overview)
- [User manual :>](#user_manual)
  * [1. Applications and Packages that we use to run my project](#applications_and_packages)
  * [2. Test my project](#test_my_prj)

### Overview
In this project, I create an image compression application using the "customtkinter" library. I use four different image compression algorithms: PCA, FFT, Wavelet, and SVD.
Since the image compression algorithms are still in the early stages of development, you are free to modify or manage them as you see fit. The only thing we need to worry about in this situation is how the app functions.

### User manual :>
#### 1. Applications and Packages that we use to run my project
  - **Applications:** _Visual Studio Code_ used with _Python 3.11 (3.11.7)_ installed from Microsoft Store.
  - **Packages:**
      1. **CustomTkinter** 5.2.1: ```pip install customtkinter==5.2.1```
      2. **Packaging** 23.2: ```pip install packaging==23.2```
      3. **Pillow** 10.1.0: ```pip install Pillow==10.1.0```
      4. **Numpy** 1.26.2: ```pip uninstall numpy==1.26.2```
      5. **PyWavelet** 1.5.0: ```pip install PyWavelets==1.5.0```
      6. **Sklearn** 1.3.2: ```pip install scikit-learn==1.3.2```
      7. **Matplotlib** 3.8.2: ```pip install matplotlib==3.8.2```
      8. **Scikit-image**: ```pip install scikit-image==0.22.0```
  
  Use the following command to install all of the above packages if you haven't already. ```pip install -r requirements.txt```
  #### 2. Test my project
This is application's interface:
![image](https://github.com/loihoang1411/Image_Compression_Py-App/assets/126635820/74b21b86-1abe-418e-a1ac-7bffeafa20c1)

1. **Add an image:**
![image](https://github.com/loihoang1411/Image_Compression_Py-App/assets/126635820/555cf863-f8c6-45b9-8d21-7d9246ad8517)

2. **Data input:** SVD rank, FFT keep, Wavelet keep, PCA component:
![image](https://github.com/loihoang1411/Image_Compression_Py-App/assets/126635820/be37f455-83b4-4895-bdef-95a16ca9b329)

  #### More detail:
  - I used to use .jpg or .png files.
  - **Width and Height** are the picture's dimensions on both sides.
  - **Size on disk** is the image's size as it is saved to a computer.
  - **Compession Ratio** is the ratio of the compressed image's size to its original size.
  - **Compression Factor** is the ratio of the image's size between its original and compressed sizes.
  - **MSE** stands for Mean Squared Error,the lower its value, the better the image quality and conversely, the higher its value, the poorer the image quality.
  - **PSNR** stands for Peak Signal-to-Noise Ratio, the higher its value, the better the image quality and conversely, the lower its value, the poorer the image quality.
  - **Compression Time** is the duration of image compression, the amount of time needed for the app to complete its full operation.
