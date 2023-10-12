Small educational project written in Python. 

Consists of two parts: 

"server" - application, which reads images from camera and recognizes colors. When one of preset colors is recognized, corresponding image shows

"client" - application, which cycles red, green and blue images when button is pressed

Image recognition 

Model is a .npy file, containing numpy-array with shape of color space. 

Color space dimensions can be configured in config.py file.

Training can be done with my object_recognition library, which I will release someday as a separate project (I hope) :)

Compiled with PyInstaller
![image](https://github.com/UltraGreed/color_receiver/assets/35086784/55fdeff5-6ab7-4102-9d77-da50a85f657b)
![image](https://github.com/UltraGreed/color_receiver/assets/35086784/bf964bc6-ed3a-4582-aab0-c6271fcc01d9)
