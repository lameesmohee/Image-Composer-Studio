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
from skimage.transform import  resize
from matplotlib.widgets import RectangleSelector
from collections import Counter



MainUI, _ = loadUiType(path.join(path.dirname(__file__), 'untitled.ui'))


class Image:
    def __init__(self,main_window):
        self.image_gray_scale = None
        self.file_path = None
        self.main_window = main_window
        self.figures = []
        self.axes =[]
        self.selected_rectangle = None
        self.counter = 0
        self.updated = False
        self.images = [None] * 6
        self.images_selected = [None] * 6
        self.selected_image = False
        self.images_mode_data = {}
        self.rs = [None] * 6
        self.scenes = []
        self.mixer = False
        self.counter_comp = 0
        self.visited_img = []
        self.image_component={
            "Magnitude": [],
            "Phase": [],
            "Real Components": [],
            "Imaginary Components": []

        }
        self.image_components = {}
        self.sliders = [self.main_window.horizontalSlider_img1,
                       self.main_window.horizontalSlider_img2,
                       self.main_window.horizontalSlider_img3,
                       self.main_window.horizontalSlider_img4]


        self.image_components_func ={
            "Magnitude": self.get_magnitude,
            "Phase": self.get_Phase,
            "Real Components" : self.get_Real,
            "Imaginary Components": self.get_Imaginary

        }

        self.create_figure()
        self.create_scene()
        self.draw_images={
            0: [self.main_window.graphicsView_img0, self.main_window.first_img_original, self.main_window.add_img1_btn],
            1: [self.main_window.graphicsView_img1, self.main_window.second_img_original, self.main_window.add_img2_btn],
            2: [self.main_window.graphicsView_img2, self.main_window.third_img_original, self.main_window.add_img3_btn],
            3: [self.main_window.graphicsView_img3,self.main_window.fourth_img_original, self.main_window.add_img4_btn],
            4: [self.main_window.output_port_1],
            5: [self.main_window.output_port_2],

        }

    def create_figure(self):
        for __ in range(0,6):
            # fig = Figure(figsize=(4, 5))
            fig = Figure(figsize=(4.5, 4.5))
            ax = fig.add_subplot()
            ax.set_position([-0.04, 0.34, 0.75, 0.65])
            self.figures.append(fig)
            self.axes.append(ax)

    def create_scene(self):
        for __ in range(0, 6):
            scene = QtWidgets.QGraphicsScene()
            self.scenes.append(scene)

    def open_image(self,file_path,button_name):
        pixmap = QPixmap(file_path)
        image = pixmap.toImage()

        if button_name != 4 and button_name != 5:
           image_data = imread(file_path)
           self.images[button_name] = image_data

        self.file_path = file_path
        self.grayscale_image = self.convert_to_grayscale(image)
        image = self.grayscale_image
        self.draw_images[button_name][1].setPixmap(QPixmap.fromImage(image).scaled(300, 300,
                                                                                   aspectRatioMode=True,
                                                                                   transformMode=1))
        self.draw_images[button_name][1].setScaledContents(True)
        if not self.draw_images[button_name][2] is None:
            self.draw_images[button_name][2].hide()




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
        return np.abs(image), np.angle(image) ,image


    def line_select_callback(self,eclick, erelease,image_index):
        print("allllo")
        if self.selected_image:
            self.selected_rectangle.remove()

        print(f"image_index:{image_index},lllllllllllllllllll")

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print(f"axes:{x1,y1,x2,y2}")
        self.selected_rectangle = plt.Rectangle(
            (min(x1,x2), min(y1,y2)),
            np.abs(x1 - x2),
            np.abs(y1 - y2),
            color='r',
            fill=False
        )
        for image_index, value in self.images_mode_data.items():
            self.images_selected[image_index] = value[1][int(min(y1, y2)): int(max(y1, y2)),
                                                int(min(x1, x2)): int(max(x1, x2))]

        self.axes[image_index].add_patch(self.selected_rectangle)

        self.selected_image = True

        self.figures[image_index].canvas.draw()

        for image_index, value in self.images_mode_data.items():
            # QCoreApplication.processEvents()
            self.components_mixer(self.sliders[image_index].value(),image_index)

        self.get_components_mixer()



    def selection(self,image_index):
        print("lama")
        if self.rs[image_index] is None:
            self.rs[image_index] = RectangleSelector(
                self.axes[image_index],
                lambda eclick, erelease: self.line_select_callback(eclick, erelease, image_index),
                useblit=False,
                button=[1],
                minspanx=5,
                minspany=5,
                spancoords="pixels",
                interactive=True,
            )
            print("RectangleSelector created")

        else:
            print("RectangleSelector already exists")

        print(self.rs)



    def plot_axes(self,mode,image_index,data):
        data = self.image_components_func[mode](mode,image_index,data)
        self.axes[image_index].imshow(data, cmap='gray')

    def get_magnitude(self, mode, image_index, image_gray_scale):
        Magnitude, phase, image = self.fourier_transform(image_gray_scale)
        self.images_mode_data[image_index] = [mode, Magnitude]
        return np.log(Magnitude) / 20

    def get_Phase(self, mode, image_index, image_gray_scale):
        Magnitude, phase, image = self.fourier_transform(image_gray_scale)
        self.images_mode_data[image_index] = [mode, phase]
        return phase

    def get_Real(self, mode, image_index, image_gray_scale):
        Magnitude, phase, image = self.fourier_transform(image_gray_scale)
        self.images_mode_data[image_index] = [mode, image.real]
        return image.real

    def get_Imaginary(self, mode, image_index, image_gray_scale):
        Magnitude, phase, image = self.fourier_transform(image_gray_scale)
        self.images_mode_data[image_index] = [mode,  image.imag]
        return image.imag

    def read_image(self,mode,image_index):
        self.axes[image_index].cla()
        print(f"mode:{mode}")
        self.image_gray_scale = rgb2gray(self.images[image_index])
        self.image_gray_scale = resize(self.image_gray_scale,(300,300),anti_aliasing=True)
        self.plot_axes(mode,image_index,self.image_gray_scale)
        self.update_graph(image_index)
        self.figures[image_index].canvas.mpl_connect(
            'button_press_event',
            lambda event, index=image_index: self.selection(index)
        )
        # self.figures[image_index].canvas.mpl_connect('button_press_event',self.selection)
        # if len(self.images_mode_data) > 1:
        #     self.get_components_mixer()

    def update_graph(self,image_index):
        print(f"inde:{image_index}")
        self.counter += 1
        print(f"count:{self.counter}")
        self.updated = True
        self.axes[image_index].set_axis_off()
        self.figures[image_index].canvas.draw()
        canvas = FigureCanvasQTAgg(self.figures[image_index])
        self.draw_images[image_index][0].setScene(self.scenes[image_index])
        self.scenes[image_index].addWidget(canvas)
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

    def check_times_image(self,image_index):
        self.visited_img.append(image_index)
        counter_img = Counter(self.visited_img)
        return counter_img[image_index] ,len(counter_img)

    def sum_same_components_of_different_img(self,component_name):
        for item, value in self.image_components.items():
            if value[0] == component_name:
                print(f"len before :{len(self.image_component[value[0]]), value[0]}")
                self.image_component[value[0]].append(value[1])
                print(f"len after :{len(self.image_component[value[0]]), value[0]}")
        if not self.selected_image:
            self.get_components_mixer()


    def components_mixer(self, mixer_perce, image_index):
        self.mixer = True
        print(f"perc:{int(mixer_perce) / 100}")
        if self.selected_image:
            component_data = (int(mixer_perce) / 100) * self.images_selected[image_index]
            # self.selected_image = False
        else:
            component_data = (int(mixer_perce) / 100) * self.images_mode_data[image_index][1]

        component = self.images_mode_data[image_index][0]

        self.image_components[image_index] = [component, component_data]
        self.image_component[component] = []
        self.sum_same_components_of_different_img(component)

    def get_components_mixer(self):
        print(f"image:{len(self.images_mode_data)}")
        Mag_component = np.sum(self.image_component["Magnitude"], axis=0)
        Phase_component = np.sum(self.image_component["Phase"], axis=0)
        Real_component = np.sum(self.image_component["Real Components"], axis=0)
        imaginary_component = np.sum(self.image_component["Imaginary Components"], axis=0)
        print(Mag_component.size)
        if Mag_component.size != 1 or Phase_component.size != 1:
            print("hallo mixer 1")
            # print(f"exp:{np.exp(Phase_component*1j)}")
            mixer_result = Mag_component * np.exp(Phase_component*1j)
            # print(f"result:{mixer_result}")
            self.image_mixer(mixer_result, 4)
        print(f'real:{Real_component.size,imaginary_component.size}')
        if Real_component.size != 1 or imaginary_component.size != 1:
            mixer_result = Real_component + 1j*imaginary_component
            # print(f"result:{mixer_result}")
            self.image_mixer(mixer_result, 4)






    def image_mixer(self,data_mixer,output_port_index):

        self.axes[output_port_index].cla()

        image = np.fft.ifft2(data_mixer)

        # img_filtered = np.abs(image).clip(0, 255).astype(np.uint8)
        img_filtered = np.abs(image).clip(0, 255).astype(np.float_)
        self.axes[output_port_index].imshow(img_filtered,cmap='gray')
        # self.figures[output_port_index].canvas.draw()
        # print(img_filtered)

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
        self.image = Image(self)


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
        self.graphicsView_img0.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_img0.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_img1.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_img1.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_img2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_img2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_img3.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_img3.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.horizontalSlider_img1.valueChanged.connect(lambda: self.image.components_mixer(self.horizontalSlider_img1.value(),0))
        self.horizontalSlider_img2.valueChanged.connect(
            lambda: self.image.components_mixer(self.horizontalSlider_img2.value(), 1))
        self.horizontalSlider_img3.valueChanged.connect(
            lambda: self.image.components_mixer(self.horizontalSlider_img3.value(), 2))
        self.horizontalSlider_img4.valueChanged.connect(
            lambda: self.image.components_mixer(self.horizontalSlider_img4.value(), 3))
        
        # self.graphicsView_img0.mousePressEvent = lambda event: self.image.selection(0)


    def open_image(self, button_name):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)",
                                                  options=options)
        if fileName:
            print(f"button:{button_name}")
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