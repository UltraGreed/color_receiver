import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QFileDialog
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.uic import loadUi

import cv2

from object_recognition.model_class import HSVModel


class MainWindow(QMainWindow):
    images_table = {
        'red': 'images/meme1.jpg',
        'green': 'images/meme2.jpg',
        'blue': 'images/meme3.jpeg'
    }

    def __init__(self):
        super().__init__()

        loadUi('server_main.ui', self)

        self.model_paths = {
            'red': '',
            'green': '',
            'blue': ''
        }

        self.models = {
        }

        self.loadModelsButton.clicked.connect(self.open_settings)

        self.camera = cv2.VideoCapture(0)

    def set_color(self, color):
        if color:
            pixmap = QPixmap(self.images_table[color])
        else:
            pixmap = QPixmap(self.label.size())
            pixmap.fill(QColor("White"))

        # Update the image in the QLabel here
        self.label.setPixmap(pixmap)

    def open_settings(self):
        window = SettingsWindow(self.model_paths)
        window.exec()

        for color, path in window.model_paths.items():
            if path:
                self.model_paths[color] = path
                self.models[color] = HSVModel(path)
        print(self.models)

    def update_colors(self):
        result, image = self.camera.read()

        if not result:
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        found_color = None

        for color, model in self.models.items():
            if model.check_object(image):
                found_color = color
                break

        self.set_color(found_color)


class SettingsWindow(QDialog):
    def __init__(self, model_paths):
        super().__init__()

        loadUi('server_settings.ui', self)

        self.labels_map = {
            'red': self.redLabel,
            'green': self.greenLabel,
            'blue': self.blueLabel
        }

        self.model_paths = model_paths

        self.redButton.clicked.connect(lambda _: self.load_model('red'))
        self.greenButton.clicked.connect(lambda _: self.load_model('green'))
        self.blueButton.clicked.connect(lambda _: self.load_model('blue'))

        for color, label in self.labels_map.items():
            label.setText(self.model_paths[color])

        self.closeButton.clicked.connect(self.close)

    def load_model(self, color):
        self.model_paths[color] = QFileDialog.getOpenFileName(self, "Select model", '.', "Numpy array files (*.npy)")[0]

        self.labels_map[color].setText(self.model_paths[color])


def main():
    app = QApplication(sys.argv)

    main_window = MainWindow()
    main_window.show()

    while True:
        main_window.update_colors()
        QApplication.processEvents()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
