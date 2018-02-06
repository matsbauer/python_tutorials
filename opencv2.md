# OpenCV2 Introduction with Python3.6
by Mats Bauer

Image processing is an incredible area of software development. There are unlimited options when using cameras and the accuracy is incredibly much higher than using sensor systems. Examples for video and image processing are the Amazon supermarket, tracking your purchases using cameras, Tesla's autopilot system or automatic security systems, noticing violant activity and alerting the police.

To get started (I am working in Cloud9 again), we need Python3.6, Numpy and OpenCV2.

```sh
#sudo optional
sudo pip3 install opencv-python
sudo pip3 install numpy
```
Firstly we need to set up our environment. Start by creating a new folder for this project, I called OCV. You can do this in terminal by typing ```mkdir OCV```. Enter this folder and download the sample image (rights aren't mine) from [here](https://pixabay.com/p-316709/?no_redirect). Rename the image to ```car.png``` and place it in our OCV folder.

Now we start our python live editor in terminal by typing ```python3``` and importing our OpenCV package as follows. 
```python
>>> import cv2
```
Next we start by telling our application where to find our image and whether we want to import it in color(1), grayscale(0) or unchanged(-1). 