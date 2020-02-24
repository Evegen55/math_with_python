# https://www.qt.io/qt-for-python?utm_campaign=Qt%20for%20Python%202018&utm_content=73330812&utm_medium=social&utm_source=facebook
import sys
import random
from PySide2 import QtCore, QtWidgets, QtGui
# import cv2 # ImportError: /usr/lib/x86_64-linux-gnu/libQt5OpenGL.so.5: undefined symbol: _Z12qTriangulateRK11QVectorPathRK10QTransformd
import alsaaudio

class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.setFixedSize(800, 600)

        self.hello = ["Hallo Welt", "你好，世界", "Hei maailma",
            "Hola Mundo", "Привет мир"]
        self.audioMixer = alsaaudio.Mixer()

        self.button = QtWidgets.QPushButton("Click me!")

        self.audioSlider = QtWidgets.QSlider()
        self.audioSlider.setMinimum(0)
        self.audioSlider.setMaximum(100)
        self.audioSlider.setSliderPosition(self.audioMixer.getvolume()[0])
        self.audioSlider.setOrientation(QtCore.Qt.Vertical)
        self.audioSlider.valueChanged.connect(self.changeSystemVolume)

        self.text = QtWidgets.QLabel("Hello World")
        self.text.setAlignment(QtCore.Qt.AlignCenter)

        # Create a button
        self.buttonExit = QtWidgets.QPushButton('Exit')
        # Connect the button "clicked" signal to the exit() method
        # that finishes the QApplication
        self.buttonExit.clicked.connect(app.exit)
        self.button.clicked.connect(self.magic)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.audioSlider)
        self.layout.addWidget(self.text)
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.buttonExit)
        self.setLayout(self.layout)


    def magic(self):
        self.text.setText(random.choice(self.hello))
        # print(cv2.__version__)

    def changeSystemVolume(self):
        self.audioMixer.setvolume(self.audioSlider.value())  # Sets volume for both channels
        current_volume = self.audioMixer.getvolume()
        self.text.setText('both channels volume ' + str(current_volume[0]) + ', ' + str(current_volume[1]))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    widget = MyWidget()
    widget.show()

    sys.exit(app.exec_())