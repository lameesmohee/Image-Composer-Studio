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
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread ,imshow
from skimage.exposure import adjust_gamma
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import  Figure
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv


MainUI, _ = loadUiType(path.join(path.dirname(__file__), 'untitled.ui'))


class Image:
    def __init__(self,file_path,main_window):
        self.image_gray_scale = None
        self.file_path = file_path
        self.main_window = main_window
        self.figures = []
        self.axes =[]
        self.counter = 0
        self.updated = False
        self.images = []
        self.images_mode_data = {}


    def create_figure(self):
        for __ in range(0,6):
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot()
            self.figures.append(fig)
            self.axes.append(ax)



    def open_image(self,file_path,button_name):
        pixmap = QPixmap(file_path)
        image = pixmap.toImage()
        if not self.updated:
            self.grayscale_image = self.convert_to_grayscale(image)
            image = self.grayscale_image




        if button_name == 0:
            print("kkkk")
            self.main_window.first_img_original.setPixmap(QPixmap.fromImage(image).scaled(350, 350,
                                                                                        aspectRatioMode=True,
                                                                                        transformMode=1))
            self.main_window.first_img_original.setScaledContents(True)
            self.main_window.add_img1_btn.hide()
            # self.first_img_original.setAlignment(Qt.AlignCenter)  # Align center
        elif button_name == 1:
            self.main_window.second_img_original.setPixmap(QPixmap.fromImage(image).scaled(350, 350,
                                                                                         aspectRatioMode=True,
                                                                                         transformMode=1))
            self.main_window.second_img_original.setScaledContents(True)
            self.main_window.add_img2_btn.hide()
        elif button_name == 2:
            self.main_window.third_img_original.setPixmap(QPixmap.fromImage(image).scaled(350, 350,
                                                                                        aspectRatioMode=True,
                                                                                        transformMode=1))
            self.main_window.third_img_original.setScaledContents(True)
            self.main_window.add_img3_btn.hide()
        elif button_name == 3:
            self.main_window.fourth_img_original.setPixmap(QPixmap.fromImage(image).scaled(350, 350,
                                                                                         aspectRatioMode=True,
                                                                                         transformMode=1))
            self.main_window.fourth_img_original.setScaledContents(True)
            self.main_window.add_img4_btn.hide()

        self.create_figure()

    def convert_to_grayscale(self, image):
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

    def fourier_transform(self,image):
        image = np.fft.fftshift(np.fft.fft2(image))
        return np.log(np.abs(image)), np.angle(image) ,image


    def get_mode_for_each_image(self,image_index):
        if image_index == 0:
          mode = self.main_window.comboBox_img1.currentText()
        elif image_index == 1:
            mode = self.main_window.comboBox_img2.currentText()
        elif image_index == 2:
            mode = self.main_window.comboBox_img3.currentText()
        elif image_index == 3:
            mode = self.main_window.comboBox_img4.currentText()

        return mode



    def read_image(self,mode,image_index):
        self.axes[image_index].cla()
        image_data = imread(self.file_path)
        self.image_gray_scale = rgb2gray(image_data)
        # self.images[image_index].append(self.image_gray_scale)
        Magnitude, phase, image= self.fourier_transform(self.image_gray_scale)
        if mode == 'Magnitude':
            self.axes[image_index].imshow(Magnitude,cmap='Grays')
            self.images_mode_data[image_index] = [mode,Magnitude]

        elif mode == 'Real Components':
            self.axes[image_index].imshow(image.real, cmap='Grays')
            self.images_mode_data[image_index] = [mode, image.real]

        elif mode =="Phase":
            self.axes[image_index].imshow(phase, cmap='Grays')
            self.images_mode_data[image_index] = [mode, phase]

        elif mode == "Imaginary Components":
            self.axes[image_index].imshow(image.imag,cmap="Grays")
            self.images_mode_data[image_index] = [mode, image.imag]

        self.update_graph(image_index)
        if len(self.images_mode_data) > 1:
            self.get_components_mixer()






    def update_graph(self,image_index):
        print(f"inde:{image_index}")
        self.counter += 1
        self.updated = True
        self.figures[image_index].canvas.draw()
        canvas = FigureCanvasQTAgg(self.figures[image_index])
        image_path= r"C:\Users\lama zakaria\Desktop\Image-Composer-Studio\image"+str(self.counter)
        self.axes[image_index].set_axis_off()
        self.figures[image_index].savefig(image_path,transparent=True, bbox_inches='tight')
        plt.close(self.figures[image_index])
        self.open_image(image_path,image_index)
        return

    def brightness_contrast(self,image,image_index,bright,contrast):
        new_image = np.zeros(image.shape,dtype=image.dtype)
        for row in image:
            for column in image:
                for channel in image:
                    new_image[row,column,channel] = np.clip(contrast*image[row,column,channel]+bright,
                                                          0,255)

        # image = adjust_gamma(image,gamma=gamma)    # less than 1 brightness  more than 1 contrast
        self.axes[image_index].imshow(new_image, cmap="Gray")
        self.update_graph(image_index)



    def get_components_mixer(self):
        Mag_component, Phase_component, Real_component, imaginary_component = 0, 0, 0, 0
        for item , value in self.images_mode_data.items():
            if value[0] == 'Magnitude':
                Mag_component = value[1]
            elif value[0] == "Phase":
                Phase_component = np.exp(value[1]* 1j)
            elif value[0] == "Real Components":
                Real_component = value[1]
            else:
                imaginary_component = value[1]

        if Mag_component and Phase_component:
            mixer_result = Mag_component * Phase_component
            self.image_mixer(mixer_result, 4)

        if Real_component and imaginary_component:
            mixer_result = Real_component + imaginary_component
            self.image_mixer(mixer_result, 4)









    def image_mixer(self,data_mixer,output_port_index):
        image = np.fft.ifft2(data_mixer)

        self.axes[output_port_index].imshow(image,cmap='Gray')
        self.figures[output_port_index].canvas.draw()
        self.update_graph(output_port_index)
        return


        # label.setPixmap(canvas.get_default_renderer().toPixmap())












class MainApp(QMainWindow, MainUI):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.handle_pushbuttons()
        self.a = "hallo"



    def handle_pushbuttons(self):
        self.add_img1_btn.clicked.connect(lambda: self.open_image(0))
        self.add_img2_btn.clicked.connect(lambda: self.open_image(1))
        self.add_img3_btn.clicked.connect(lambda: self.open_image(2))
        self.add_img4_btn.clicked.connect(lambda: self.open_image(3))
        self.comboBox_img1.currentTextChanged.connect(lambda: self.image.read_image(self.comboBox_img1.currentText(),0))
        self.comboBox_img2.currentTextChanged.connect(
            lambda: self.image.read_image(self.comboBox_img2.currentText(), 1))
        self.comboBox_img3.currentTextChanged.connect(
            lambda: self.image.read_image(self.comboBox_img3.currentText(), 2))
        self.comboBox_img4.currentTextChanged.connect(
            lambda: self.image.read_image(self.comboBox_img4.currentText(), 3))


    def open_image(self, button_name):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)",
                                                  options=options)
        if fileName:
            print(f"button:{button_name}")
            self.image = Image(fileName, self)

            self.image.open_image(fileName,button_name)


def main():
    app = QApplication(sys.argv)
    window = MainApp()
    # image = Image(None)
    # image.main_window.show()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()