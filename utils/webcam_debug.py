import tkinter

from PIL import Image, ImageTk

import cv2

from object_recognition.model_class import HSVModel, get_model_path

import time

COLORS = "red", "green", "blue"
MODELS = {
    "red": HSVModel(f'object_recognition/{get_model_path("sub", "red")}'),
    "green": HSVModel(f'object_recognition/{get_model_path("sub", "green")}'),
    "blue": HSVModel(f'object_recognition/{get_model_path("sub", "blue")}')
}

n_cols = 2
n_rows = 3

root = tkinter.Tk()
root.title('Imaginarium')

label_widgets = dict()

is_resizing = False

camera = cv2.VideoCapture(0)

for i, color in enumerate(COLORS):
    result, image = camera.read()

    MODELS[color].image = image
    grayscale = MODELS[color].get_grayscale()

    label_widgets[color] = tkinter.Label(root, image=ImageTk.PhotoImage(Image.fromarray(image).resize((360, 360))))
    label_widgets[f"{color}_gs"] = tkinter.Label(root, image=ImageTk.PhotoImage(Image.fromarray(grayscale).resize((360, 360))))

    label_widgets[color].grid(row=i, column=0, sticky='nsew')
    label_widgets[f"{color}_gs"].grid(row=i, column=1, sticky='nsew')


# Update plots in infinite loop.
while True:
    result, image = camera.read()
    cv2.imwrite("test_img1.png", image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # New message has come.
    for i, color in enumerate(COLORS):
        MODELS[color].image = image
        grayscale = MODELS[color].get_grayscale()

        image_tk = ImageTk.PhotoImage(Image.fromarray(image).resize((360, 360)))
        label_widgets[color].config(image=image_tk)
        label_widgets[color].image = image_tk

        grayscale_tk = ImageTk.PhotoImage(Image.fromarray(grayscale).resize((360, 360)))
        label_widgets[f"{color}_gs"].config(image=grayscale_tk)
        label_widgets[f"{color}_gs"].image = grayscale_tk

    root.update()
    root.update_idletasks()