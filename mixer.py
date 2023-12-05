import math
import os
import matplotlib
matplotlib.use('Qt5Agg')
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUiType
from os import path
import sys
from qtawesome import icon
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap, QImage, qRgb, QColor

MainUI, _ = loadUiType(path.join(path.dirname(__file__), 'untitled.ui'))


class MainApp(QMainWindow, MainUI):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.handle_pushbuttons()

    def handle_pushbuttons(self):
        self.add_img1_btn.clicked.connect(lambda: self.open_image(self.add_img1_btn))
        self.add_img2_btn.clicked.connect(lambda: self.open_image(self.add_img2_btn))
        self.add_img3_btn.clicked.connect(lambda: self.open_image(self.add_img3_btn))
        self.add_img4_btn.clicked.connect(lambda: self.open_image(self.add_img4_btn))


    def open_image(self, button_name):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)",
                                                  options=options)
        if fileName:
            pixmap = QPixmap(fileName)
            image = pixmap.toImage()
            grayscale_image = self.convert_to_grayscale(image)
            if button_name == self.add_img1_btn:
                self.first_img_original.setPixmap(QPixmap.fromImage(grayscale_image).scaled(350, 350,
                                                                         aspectRatioMode=True, transformMode=1))
                self.first_img_original.setScaledContents(True)
                self.add_img1_btn.hide()
                # self.first_img_original.setAlignment(Qt.AlignCenter)  # Align center
            elif button_name == self.add_img2_btn:
                self.second_img_original.setPixmap(QPixmap.fromImage(grayscale_image).scaled(350, 350,
                                                                         aspectRatioMode=True, transformMode=1))
                self.second_img_original.setScaledContents(True)
                self.add_img2_btn.hide()
            elif button_name == self.add_img3_btn:
                self.third_img_original.setPixmap(QPixmap.fromImage(grayscale_image).scaled(350, 350,
                                                                         aspectRatioMode=True, transformMode=1))
                self.third_img_original.setScaledContents(True)
                self.add_img3_btn.hide()
            elif button_name == self.add_img4_btn:
                self.fourth_img_original.setPixmap(QPixmap.fromImage(grayscale_image).scaled(350, 350,
                                                                         aspectRatioMode=True, transformMode=1))
                self.fourth_img_original.setScaledContents(True)
                self.add_img4_btn.hide()

    def convert_to_grayscale(self,image):
        # Retrieving the width and height of the image to loop through all pixels.
        width = image.width()
        height = image.height()

        for y in range(height):
            for x in range(width):
                pixel = image.pixel(x, y)
                r, g, b, _ = QColor(pixel).getRgb()
                # Formula for converting a color image to grayscale
                gray = int(0.299 * r + 0.587 * g + 0.114 * b)
                gray_color = qRgb(gray, gray, gray)
                image.setPixel(x, y, gray_color)

        return image


def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()