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
import cv2



MainUI, _ = loadUiType(path.join(path.dirname(__file__), 'untitled.ui'))


class Image:
    def __init__(self,main_window):
        self.output_port = 4
        self.image_gray_scale = None
        self.file_path = None
        self.main_window = main_window
        self.figures = []
        self.axes =[]
        # self.selected_rectangle = None
        self.counter = 0
        self.updated = False
        self.images = [None] * 6
        self.images_selected = [None] * 6
        self.selected_image = False
        self.images_mode_data = {}
        self.rs = [None] * 6
        self.selected_rectangle = [None] * 6
        self.scenes = []
        self.mixer = False
        self.x1, self.y1 = 0, 0
        self.x2, self.y2 = 0, 0
        self.counter_comp = 0
        self.visited_img = []
        self.labels_img = [self.main_window.img_label1,
                           self.main_window.img_label2,
                           self.main_window.img_label3,
                           self.main_window.img_label4
                           ]

        self.image_component={
            "Magnitude": [],
            "Phase": [],
            "Real Components": [],
            "Imaginary Components": []
        }

        self.mag_and_phase = False
        self.real_and_imaginary = False
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
        self.selected_index = 0
        self.brightness = 0
        self.contrast = 1.0

        # Connect mouse events for brightness adjustment
        self.main_window.first_img_original.mousePressEvent = self.mousePressEvent
        self.main_window.second_img_original.mousePressEvent = self.mousePressEvent
        self.main_window.third_img_original.mousePressEvent = self.mousePressEvent
        self.main_window.fourth_img_original.mousePressEvent = self.mousePressEvent
        self.main_window.first_img_original.mouseMoveEvent = self.mouseMoveEvent
        self.main_window.second_img_original.mouseMoveEvent = self.mouseMoveEvent
        self.main_window.third_img_original.mouseMoveEvent = self.mouseMoveEvent
        self.main_window.fourth_img_original.mouseMoveEvent = self.mouseMoveEvent


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
        # Checking if the image isn't an output image
        if button_name != 4 and button_name != 5:
           # Getting the image data and putting the images into a dictionary for easy access
           image_data = imread(file_path)
           self.images[button_name] = image_data

        self.file_path = file_path
        #Converting the image into grayscale
        self.grayscale_image = self.convert_to_grayscale(image)
        image = self.grayscale_image
        # Viewing the Image in it's viewport and setting it's size to 300 * 300
        self.draw_images[button_name][1].setPixmap(QPixmap.fromImage(image).scaled(300, 300,
                                                                                   aspectRatioMode=True,
                                                                                   transformMode=1))
        self.draw_images[button_name][1].setScaledContents(True)
        # Checking if an Image exist to remove the button if an image was found
        if not self.draw_images[button_name][2] is None:
            self.draw_images[button_name][2].hide()

    def convert_to_grayscale(self, image):
        # Retrieving the width and height of the image to loop through all pixels.
        width = image.width()
        height = image.height()
        # Looping around each pixel
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
        # FT function to get the magnitude, phase, real and imaginary components of an image
        image = np.fft.fftshift(np.fft.fft2(image))
        return np.abs(image), np.angle(image) ,image

    def get_selected_image(self):
        for image_index, value in self.images_mode_data.items():
            self.selected_rectangle[image_index] = plt.Rectangle(
                (min(self.x1, self.x2), min(self.y1, self.y2)),
                np.abs(self.x1 - self.x2),
                np.abs(self.y1 - self.y2),
                color='r',
                fill=False
            )
            self.axes[image_index].add_patch(self.selected_rectangle[image_index])
            self.figures[image_index].canvas.draw()

            self.images_selected[image_index] = value[1][int(min(self.y1, self.y2)): int(max(self.y1, self.y2)),
                                                int(min(self.x1, self.x2)): int(max(self.x1, self.x2))]
            print(f"lenght_of_selected_image:{len(self.images_selected[image_index])}")
            if len(self.images_selected[image_index]) == 0:
                self.selected_image = False
            else:
                self.selected_image = True
            if self.selected_image:
                self.selected_rectangle[image_index].remove()


    def line_select_callback(self,eclick, erelease,image_index):
        print("allllo")


        print(f"image_index:{image_index},lllllllllllllllllll")

        self.x1, self.y1 = eclick.xdata, eclick.ydata
        self.x2, self.y2 = erelease.xdata, erelease.ydata

        self.get_selected_image()

        # self.axes[image_index].add_patch(self.selected_rectangle)

        # for image_index, value in self.images_mode_data.items():
            # QCoreApplication.processEvents()
        self.components_mixer(self.sliders[image_index].value(),image_index)
        # self.get_components_mixer()


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
        # Function to plot the Chosen component for an image
        self.axes[4].cla()
        self.axes[5].cla()
        self.figures[4].canvas.draw()
        self.figures[5].canvas.draw()
        data = self.image_components_func[mode](mode,image_index,data)
        self.axes[image_index].imshow(data, cmap='gray')

    def get_magnitude(self, mode, image_index, image_gray_scale):
        Magnitude, phase, image = self.fourier_transform(image_gray_scale)
        self.images_mode_data[image_index] = []
        self.images_mode_data[image_index] = [mode, Magnitude]
        return np.log(Magnitude) / 20

    def get_Phase(self, mode, image_index, image_gray_scale):
        Magnitude, phase, image = self.fourier_transform(image_gray_scale)
        self.images_mode_data[image_index] = []
        self.images_mode_data[image_index] = [mode, phase]
        return phase

    def get_Real(self, mode, image_index, image_gray_scale):
        Magnitude, phase, image = self.fourier_transform(image_gray_scale)
        self.images_mode_data[image_index] = []
        self.images_mode_data[image_index] = [mode, image.real]
        return image.real

    def get_Imaginary(self, mode, image_index, image_gray_scale):
        Magnitude, phase, image = self.fourier_transform(image_gray_scale)
        self.images_mode_data[image_index] = []
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
        # Checking whether there's a selected rectangle or not
        if self.selected_image:
            self.get_selected_image()
        if len(self.images_mode_data) > 1:
            self.components_mixer(1,1)

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

    def check_times_image(self,image_index):
        self.visited_img.append(image_index)
        counter_img = Counter(self.visited_img)
        return counter_img[image_index] ,len(counter_img)
    
    def initial_declare(self):
        self.image_component = {
            "Magnitude": [],
            "Phase": [],
            "Real Components": [],
            "Imaginary Components": []
        }

    def sum_same_components_of_different_img(self):
        self.initial_declare()
        for item, value in self.image_components.items():
            # self.image_component[value[0]] = []
            print(f"len before :{len(self.image_component[value[0]]), value[0]}")
            self.image_component[value[0]].append(value[1])
            print(f"len after :{self.image_component[value[0]], value[0]}")

        # if not self.selected_image:
        self.get_components_mixer()

    def select_inner_outer_corner(self,image_index,case):
        if case == 'inner':
            self.images_selected[image_index] = self.images_mode_data[image_index][1]
        else:
            self.images_selected[image_index] = self.images_mode_data[image_index][1] - self.images_selected[image_index]


    def get_data(self):
        for item, value in self.images_mode_data.items():
            component = value[0]
            image_index = item
            self.labels_img[image_index].setText(str(self.sliders[item].value())+"%")
            # self.select_inner_outer_corner(image_index,case)



            if self.selected_image:
                component_data = (int(self.sliders[item].value()) / 100) * self.images_selected[image_index]
                print(f"selected_imag:{self.images_selected[image_index]}")
            else:
                component_data = (int(self.sliders[item].value()) / 100) * self.images_mode_data[image_index][1]
            self.image_components[image_index] = [component, component_data]
                # self.get_data(self.images_selected[image_index])
                # component_data = (int(mixer_perce) / 100) * self.images_mode_data[image_index][1]


    def components_mixer(self, mixer_perce, image_index):
        self.mixer = True
        # print(f"perc:{int(mixer_perce) / 100}")
        self.get_data()
        # if self.selected_image:
        #     self.get_data(self.images_selected[image_index])
        #     # component_data = (int(mixer_perce) / 100) * self.images_selected[image_index]
        #     # self.selected_image = False
        # else:
        #     self.get_data(self.images_selected[image_index])
        #     # component_data = (int(mixer_perce) / 100) * self.images_mode_data[image_index][1]

        # component = self.images_mode_data[image_index][0]
        #
        # self.image_components[image_index] = [component, component_data]

        self.sum_same_components_of_different_img()

    def check_output_port(self,output_port):
        self.output_port = output_port

    def which_case_is_mixer(self,case):
        if case:
            if self.output_port == 5:
                output_port = 4
            else:
                output_port = 5
        else:
            output_port = self.output_port

        return output_port


    def get_components_mixer(self):
        print(f"image_mode_data:{len(self.images_mode_data)}")
        Mag_component = np.sum(self.image_component["Magnitude"], axis=0)
        Phase_component = np.sum(self.image_component["Phase"], axis=0)
        Real_component = np.sum(self.image_component["Real Components"], axis=0)
        imaginary_component = np.sum(self.image_component["Imaginary Components"], axis=0)
        print(f"Mag:{Mag_component.size}")
        print(f"MAG_data:{Mag_component}")
        if Mag_component.size > 1 or Phase_component.size > 1:
            self.mag_and_phase = True
            print("hallo mixer 1")
            mixer_result = Mag_component * np.exp(Phase_component*1j)
            # output_port = self.which_case_is_mixer(self.real_and_imaginary)
            self.image_mixer(mixer_result, 4)
        print(f'real:{Real_component.size,imaginary_component.size}')


        if Real_component.size > 1 or imaginary_component.size > 1:
            self.real_and_imaginary = True
            mixer_result = Real_component + 1j*imaginary_component
            # print(f"result:{mixer_result}")
            # output_port = self.which_case_is_mixer(self.mag_and_phase)
            self.image_mixer(mixer_result,  5)

    def image_mixer(self,data_mixer,output_port_index):
        self.axes[output_port_index].cla()
        image = np.fft.ifft2(data_mixer)
        # img_filtered = np.abs(image).clip(0, 255).astype(np.uint8)
        img_filtered = np.abs(image).clip(0, 255).astype(np.float_)
        self.axes[output_port_index].imshow(img_filtered,cmap='gray')
        # self.figures[output_port_index].canvas.draw()
        self.update_graph(output_port_index)
        return
        # label.setPixmap(canvas.get_default_renderer().toPixmap())
    
    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.lastPos = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.RightButton:
            dy = event.y() - self.lastPos.y()
            if dy > 0:
                self.brightness += 1  # Increase brightness when moving up
            else:
                self.brightness -= 1  # Decrease brightness when moving down

            dx = event.x() - self.lastPos.x()
            if dx > 0:
                self.contrast += 0.1  # Increase contrast when moving right
            else:
                self.contrast -= 0.1  # Decrease contrast when moving left

            self.lastPos = event.pos()
            self.paint_brightness_contrast_adjusted_image(self.selected_index)

    def paint_brightness_contrast_adjusted_image(self, image_index):
        if self.images[image_index] is None:
            return

        img = self.images[image_index].copy()  

        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # Convert the image to grayscale

        adjusted = cv2.convertScaleAbs(img, alpha=self.contrast, beta=self.brightness)

        height, width = adjusted.shape
        bytes_per_line = width
        q_img = QImage(adjusted.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(q_img)

        label = self.draw_images[image_index][1] 
        label.setPixmap(pixmap)
        label.setScaledContents(True)

        add_image_button = self.draw_images[image_index][2]
        if add_image_button is not None:
            add_image_button.hide()

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
        self.output_port_1.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.output_port_1.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.output_port_1.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.output_port_1.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.output_port_2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.output_port_2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.horizontalSlider_img1.valueChanged.connect(lambda: self.image.components_mixer(self.horizontalSlider_img1.value(),0))
        self.horizontalSlider_img2.valueChanged.connect(
            lambda: self.image.components_mixer(self.horizontalSlider_img2.value(), 1))
        self.horizontalSlider_img3.valueChanged.connect(
            lambda: self.image.components_mixer(self.horizontalSlider_img3.value(), 2))
        self.horizontalSlider_img4.valueChanged.connect(
            lambda: self.image.components_mixer(self.horizontalSlider_img4.value(), 3))
        self.radioButton_output_port1.clicked.connect(lambda : self.image.check_output_port(4))
        self.radioButton_output_port2.clicked.connect(lambda: self.image.check_output_port(5))
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
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()