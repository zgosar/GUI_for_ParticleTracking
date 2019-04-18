import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QWidget, QLabel, QLineEdit, 
    QTextEdit, QGridLayout, QApplication,
    QSlider, QVBoxLayout, QHBoxLayout,
    QSpacerItem, QSizePolicy, QCheckBox)

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
        self.x = self.slider.value()
        if self.integer:
            return int(self.x)
        return self.x

    def setLabelValue(self, value=''):
        if value=='': value = self.getValue()
        self.x = value #self.minimum + (float(value) / (self.slider.maximum() - self.slider.minimum())) * (
            #self.maximum - self.minimum)
        if self.integer:
            self.x = int(self.x)
        self.label.setText(self.label_format.format(self.x))

    def setSliderPosition(self, value):
        if value is not None:
            self.slider.setSliderPosition(value)

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
        self.x = 0
        self.setLabelValue(self.slider.value())

    def getValue(self):
        #print("VS getValue", self.slider.value())
        self.x = self.slider.value()
        if self.integer:
            return int(self.x)
        return self.x

    def setLabelValue(self, value=''):
        #print("VS setValue", self.slider.value())
        if value=='': value = self.getValue()
        self.x = value #self.minimum + (float(value) / (self.slider.maximum() - self.slider.minimum())) * (
            #self.maximum - self.minimum)
        if self.integer:
            self.x = int(self.x)
            self.label.setText("{:}".format(self.x))
        else:
            self.label.setText("{:.4f}".format(self.x))

    def setSliderPosition(self, value):
        if value is not None:
            self.slider.setSliderPosition(value)

class DiameterVerticalSlider(VerticalSlider):

    def __init__(self, minimum, maximum,
                 integer=False, top=True,
                 title='', parent=None):
        VerticalSlider.__init__(self, (minimum-1)//2, (maximum-1)//2,
                 integer=integer, top=top,
                 title=title, parent=parent)

        self.slider.valueChanged.disconnect(self.setLabelValue)
        self.slider.valueChanged.connect(self.setLabelValue)

    def getValue(self):
        self.x = 2*self.slider.value()+1
        return self.x

    def setLabelValue(self, value=''):
        #import traceback
        #traceback.print_stack()
        #print("DiameterVerticalSlider.setLabelValue", value, self.x)
        #if value=='': value = self.getValue()
        self.x = self.getValue()
        if self.integer:
            self.x = int(self.x)
            self.label.setText("{:}".format(self.x))
        else:
            self.label.setText("{:.4f}".format(self.x))
    def setSliderPosition(self, value):
        self.slider.setSliderPosition((value-1)//2)

class SeparationWidget(QWidget):
    def __init__(self, minimum, maximum,
                 integer=False, top=True,
                 title='', parent=None):
        super(SeparationWidget, self).__init__(parent=parent)
        self.verticalLayout = QVBoxLayout(self)
        
        self.label = QLabel(self)
        self.title = QLabel(self)
        self.title.setText(title)
        self.slider = QSlider(self)
        self.slider.setOrientation(Qt.Vertical)

        self.checkbox = QCheckBox("")
        self.checkbox.setChecked(False)
        #self.checkbox.stateChanged.connect(self.update_enable_state)
        # must be connected to global update
        self.checkbox_enabled = True

        self.verticalLayout.addWidget(self.title)
        self.verticalLayout.addWidget(self.checkbox)
        self.verticalLayout.addWidget(self.slider)
        self.verticalLayout.addWidget(self.label)

        self.slider.setTickInterval(1)
        self.slider.setSingleStep(1)
        self.slider.setValue((minimum+maximum)/2)

        self.slider.setMinimum(minimum)
        self.slider.setMaximum(maximum)
        self.integer = integer
        self.slider.valueChanged.connect(self.valueChangedExecute)
        self.x = None
        self.setLabelValue(self.slider.value())

    def update_enable_state(self):
        if self.checkbox.isChecked():
            self.checkbox.setText("")
            self.checkbox_enabled = True
            self.setLabelValue(self.getValue())

        else:
            self.checkbox.setText("")
            self.checkbox_enabled = False
            self.label.setText("Default\n({:})")

    def getValue(self):
        self.x = self.slider.value()
        if self.integer:
            return int(self.x)
        return self.x

    def setLabelValue(self, value=''):
        if self.checkbox_enabled:
            if value=='': value = self.getValue()
            self.x = int(value)
            self.label.setText("{:}".format(self.x))
        else:
            self.x = int(value)
            self.label.setText("Default\n({:})".format(value))

    def valueChangedExecute(self):
        if self.checkbox_enabled:
            value = self.getValue()
            self.setLabelValue(value)
            self.x = value
        else:
            pass
            
            #self.setSliderPosition(self.x)

    def setSliderPosition(self, value1, diameter=''):
        if self.checkbox_enabled:
            self.slider.setSliderPosition(value1)
        else:
            self.slider.setSliderPosition(diameter+1)
            self.setLabelValue(self.getValue())
