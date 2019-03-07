import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QWidget, QLabel, QLineEdit, 
    QTextEdit, QGridLayout, QApplication,
    QSlider, QVBoxLayout, QHBoxLayout,
    QSpacerItem, QSizePolicy)

class HorizontalSlider(QWidget):
    def __init__(self, minimum, maximum, integer=False, left=True, parent=None, label_format=''):
        super(HorizontalSlider, self).__init__(parent=parent)
        self.horizontalLayout = QHBoxLayout(self)
        
        self.label = QLabel(self)
        self.slider = QSlider(self)
        self.slider.setOrientation(Qt.Horizontal)
        
        if left:
            self.horizontalLayout.addWidget(self.label)
            self.horizontalLayout.addWidget(self.slider)
        else:
            self.horizontalLayout.addWidget(self.slider)
            self.horizontalLayout.addWidget(self.label)

        self.slider.setMinimum(minimum)
        self.slider.setMaximum(maximum)
        self.integer = integer
        self.slider.valueChanged.connect(self.setLabelValue)
        self.x = None
        self.slider.setSingleStep(1)
        #self.slider.setTickInterval(1)

        if label_format == '':
            if self.integer:
                label_format = "{:}"
            else:
                label_format = "{:.4f}"
        self.label_format = label_format

        self.setLabelValue(self.slider.value())

    def getValue(self):
        x = self.slider.value()
        if self.integer:
            return int(self.x)
        return x

    def setLabelValue(self, value=''):
        if value=='': value = self.getValue()
        self.x = value #self.minimum + (float(value) / (self.slider.maximum() - self.slider.minimum())) * (
            #self.maximum - self.minimum)
        if self.integer:
            self.x = int(self.x)
        self.label.setText(self.label_format.format(self.x))

class VerticalSlider(QWidget):
    def __init__(self, minimum, maximum,
                 integer=False, top=True,
                 title='', parent=None):
        super(VerticalSlider, self).__init__(parent=parent)
        self.verticalLayout = QVBoxLayout(self)
        
        self.label = QLabel(self)
        self.title = QLabel(self)
        self.title.setText(title)
        self.slider = QSlider(self)
        self.slider.setOrientation(Qt.Vertical)
        if top:
            self.verticalLayout.addWidget(self.title)
            self.verticalLayout.addWidget(self.label)
            self.verticalLayout.addWidget(self.slider)
        else:
            self.verticalLayout.addWidget(self.title)
            self.verticalLayout.addWidget(self.slider)
            self.verticalLayout.addWidget(self.label)
            
        #self.resize(self.sizeHint())
        self.slider.setTickInterval(1)
        self.slider.setSingleStep(1)
        self.slider.setValue((minimum+maximum)/2)

        self.slider.setMinimum(minimum)
        self.slider.setMaximum(maximum)
        self.integer = integer
        self.slider.valueChanged.connect(self.setLabelValue)
        self.x = None
        self.setLabelValue(self.slider.value())

    def getValue(self):
        x = self.slider.value()
        if self.integer:
            return int(self.x)
        return x

    def setLabelValue(self, value=''):
        if value=='': value = self.getValue()
        self.x = value #self.minimum + (float(value) / (self.slider.maximum() - self.slider.minimum())) * (
            #self.maximum - self.minimum)
        if self.integer:
            self.x = int(self.x)
            self.label.setText("{:}".format(self.x))
        else:
            self.label.setText("{:.4f}".format(self.x))
