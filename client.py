import sys

from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtGui import QPixmap, QColor

import itertools


class Client(QMainWindow):
    color_cycle = itertools.cycle((QColor('Red'), QColor('Green'), QColor('Blue')))

    def __init__(self):
        super().__init__()
        loadUi('client_main.ui', self)

        self.showMaximized()

        pixmap = QPixmap(self.label.size())
        pixmap.fill(QColor('White'))

        self.label.setPixmap(pixmap)

        self.pushButton.clicked.connect(self.set_random_color)

    def set_random_color(self):
        new_pixmap = QPixmap(self.label.size())
        new_pixmap.fill(next(self.color_cycle))

        self.label.setPixmap(new_pixmap)


def main():
    app = QApplication(sys.argv)
    main_window = Client()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
