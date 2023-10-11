Small educational project written in Python. 
Consists of two parts: 
    "server" - application, which reads images from camera and recognizes colors. When one of preset colors is recognized, corresponding image shows
    "client" - application, which cycles red, green and blue images when button is pressed

Image recognition 

Model is a .npy file, containing numpy-array with shape of color space. 
Color space dimensions can be configured in config.py file.
Training can be done with my object_recognition library, which I will release someday as a separate project (I hope) :)

Compiled with PyInstaller
