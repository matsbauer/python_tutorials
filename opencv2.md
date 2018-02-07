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
Next we start by telling our application where to find our image and whether we want to import it in color(1), grayscale(0) or unchanged(-1). We want to start by loading the image in grayscale, so we take the function below with the attribute 0. To have a look at the image, we need to use the cv2 function ```imwrite()``` to save a copy of our local instance ```img```.
```python
>>> img = cv2.imread('car.jpg',0)
>>> cv2.imwrite('grayscale.jpg', img)
True
```
![image converted to graysacle](https://raw.githubusercontent.com/matsbauer/python_tutorials/master/data/grayscale.jpg)
That's looking good already. For my example, I want to create a highlighting mask, to seperate the surroundings from the car and the road. We will do this, using a color filter - thus we will need the image in full color (attribute = 1), instead of in a grayscale (0).
```python
>>> img = cv2.imread('car.jpg', 1)
```
Now we have loaded the original image. For the next step, we need to find out, what color the road has. There is a simple and a more complicated (OpenCV2) way of doing it, let's start by going simple. A simple online tool for getting color codes from an image: [ColorCode Picker]("https://html-color-codes.info/colors-from-image/"). 
![Screenshot Color Picker](https://raw.githubusercontent.com/matsbauer/python_tutorials/master/data/2018-02-06_10h23_12.png)