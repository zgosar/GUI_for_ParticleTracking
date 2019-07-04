import sys
from PyQt5.QtWidgets import (QWidget, QLabel, QLineEdit, 
    QTextEdit, QGridLayout, QApplication, QVBoxLayout, QHBoxLayout,
    QSlider, QFileDialog, QPushButton, QStyle, QCheckBox, QRadioButton,
    QButtonGroup)
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSignal
import MySlider
import pyqtgraph as pg
import trackpy as tp
import pims
import TWV_Reader
import numpy as np
from time import time
from copy import deepcopy
import math
from tracking import simple_tracking, find_particles_first_frame

from batch_with_checkpointing import *

import matplotlib.pyplot as plt
viridis = plt.get_cmap('viridis')
viridis_list = viridis(np.array([np.arange(256) for i in range(2)]))
def convert_rgb_to_hex(rgb):
    return '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
def convert_float_to_color(float_list):
    return list(map(QtGui.QColor,
                        map(convert_rgb_to_hex,
                            map(viridis,
                                float_list))))

class RadioDemo(QWidget):

    def __init__(self, options, preselected, update_function, parent=None):
        super(RadioDemo, self).__init__(parent)
        self.group = QButtonGroup()
        self.options = options[:]
        self.layout = QHBoxLayout()
        self.radio_buttons = []
        for i, option in enumerate(options):
            self.radio_buttons.append(QRadioButton(option))
            self.group.addButton(self.radio_buttons[-1])
            self.radio_buttons[-1].toggled.connect(lambda value: update_function('RadioButton {:}'.format(option), value))
            self.layout.addWidget(self.radio_buttons[-1])
        self.radio_buttons[preselected].setChecked(True)
        self.last_selected = (preselected, self.options[preselected])

    def which(self):
        for i in range(len(self.radio_buttons)):
            if self.radio_buttons[i].isChecked() == True:
                self.last_selected = (i, self.options[i])
                return i, self.options[i]
        return self.last_selected # when changing, one is first unselected and the other is later selected.

class RadioDemo2(QWidget):

    def __init__(self, options, preselected, update_function, parent=None):
        super(RadioDemo2, self).__init__(parent)
        self.options = options[:]
        self.layout = QHBoxLayout()
        self.radio_buttons = []
        for i, option in enumerate(options):
            self.radio_buttons.append(QRadioButton(option))
            self.radio_buttons[-1].toggled.connect(lambda value: update_function('RadioButton {:}'.format(option), value))
            self.layout.addWidget(self.radio_buttons[-1])
        self.radio_buttons[preselected].setChecked(True)
        self.last_selected = (preselected, self.options[preselected])

    def which(self):
        for i in range(len(self.radio_buttons)):
            if self.radio_buttons[i].isChecked() == True:
                self.last_selected = (i, self.options[i])
                return i, self.options[i]
        return self.last_selected # when changing, one is first unselected and the other is later selected.
            
        
class Example(QWidget):
    gridSpacing = 10
    sig = pyqtSignal(str) # outgoing signal
    
    def __init__(self, filename):
        super().__init__()

        self.filename = filename
        self.intype = self.filename[0].split('.')[-1]
        self.running = False
        self.load_file(update=False)
        self.initUI()
        
    def init_colorbar(self):
        self.colorbarViewBox = self.win.addViewBox(lockAspect=False, enableMouse=False)
        self.linRegionUpperLimit = 10000
        self.linRegion = pg.LinearRegionItem(
            orientation=pg.LinearRegionItem.Horizontal,
            bounds=(0, 255),
            values=(self.linRegionUpperLimit//10, self.linRegionUpperLimit)
            )
        self.minmass = 90
        self.maxmass = self.linRegionUpperLimit
        self.linRegion.sigRegionChangeFinished.connect(self.on_lin_region_change)
        self.linRegion.sigRegionChanged.connect(self.on_lin_region_change)

        self.colorbar_text = []
        self.colorbar_text.append(pg.TextItem('top', anchor=(0, 0)))
        self.colorbar_text[0].setPos(0, 255)
        self.colorbar_text.append(pg.TextItem('up', anchor=(0, 0)))
        self.colorbar_text[1].setPos(0, 200)
        self.colorbar_text.append(pg.TextItem('down', anchor=(0, 0)))
        self.colorbar_text[2].setPos(0, 50)
        self.colorbar_text.append(pg.TextItem('0', anchor=(0, 0)))
        self.colorbar_text[3].setPos(0, 0)
        
        self.colorbar = pg.ImageItem(viridis_list)
        self.colorbarViewBox.addItem(self.colorbar)
        self.colorbarViewBox.addItem(self.linRegion)
        self.colorbarViewBox.setMaximumWidth(50) #resize(self.colorbarViewBox.height(), 10)
        for i in range(len(self.colorbar_text)): self.colorbarViewBox.addItem(self.colorbar_text[i])

    def load_file(self, update=False, **kwargs):
        """
        Removed as_grey kwarg handling (and any other kwargs handling for that matter.
        """
        #print(self.filename)
        if self.intype == 'png':
            self.frames = pims.ImageSequence(self.filename, as_grey=True)
        else:
            self.frames = pims.open(self.filename[0])

        #self.frames.set_end_frame(100) # DEMONSTRATION FILTERING

        self.lenframes = len(self.frames)
        self.height = len(self.frames[0])
        
        if update:
            self.fileTextbox.setText(self.filename[0])
            self.framesSlider.slider.setMaximum(self.lenframes-1)
            self.framesSlider.label_format = '{:}/' + str(self.lenframes)
            self.framesSlider.setLabelValue()
            self.particles_vs_time_plot_item.setXRange(0, self.lenframes, padding=0)
            self.particles_vs_time_plot_item_scat.clear()

            self.update('On load file update call')
        
    def initUI(self):
        self.grid = QGridLayout()
        self.grid.setSpacing(self.gridSpacing)

        self.fileSelectorLayout = QHBoxLayout()
        self.grid.addLayout(self.fileSelectorLayout, 0, 0, 1, -1)
        self.fileTextbox = QLineEdit(self)
        self.fileTextbox.setText(self.filename[0])
        self.fileSelectorLayout.addWidget(self.fileTextbox)
        self.fileButton = QPushButton('Load file')
        self.fileButton.clicked.connect(self.on_load_file)
        self.fileSelectorLayout.addWidget(self.fileButton)
        self.invertCheckbox = QCheckBox("Invert")
        self.invertCheckbox.setChecked(False)
        self.invertCheckbox.stateChanged.connect(self.update)
        self.fileSelectorLayout.addWidget(self.invertCheckbox)

        self.win = pg.GraphicsWindow(title="Basic plotting examples")
        self.grid.addWidget(self.win, 1, 0)
        self.imageImageItem = pg.ImageItem(self.frames[0])
        self.imageViewBox = self.win.addViewBox(lockAspect=True)

        self.p6 = self.imageViewBox.addItem(self.imageImageItem)
        self.imageScatter = pg.ScatterPlotItem()
        self.imageScatter.setParentItem(self.imageImageItem)

        # Start of colorbar
        self.init_colorbar()
        # End of colorbar
        
        self.framesSlider = MySlider.HorizontalSlider(0, len(self.frames)-1, integer=True, left=False,
                                                      label_format='{:}/'+str(self.lenframes))
        self.framesSliderLayout = QHBoxLayout()
        self.framesSliderLayout.addWidget(self.framesSlider)
        self.grid.addLayout(self.framesSliderLayout, 5, 0, -1, -1)
        self.framesSlider.slider.valueChanged.connect(lambda value: self.update('framesSlider', value))
        self.playButton = QPushButton('Start')
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.framesSliderLayout.addWidget(self.playButton)
        self.playButton.clicked.connect(self.start_processing)

        self.particles_vs_time_plot = pg.PlotWidget(labels={'left':'# Particles', 'bottom':'Frame'})
        self.particles_vs_time_plot_item = self.particles_vs_time_plot.getPlotItem()
        self.particles_vs_time_plot_item_scat = pg.ScatterPlotItem()
        self.particles_vs_time_plot_item.addItem(self.particles_vs_time_plot_item_scat)
        self.particles_vs_time_plot_item.setXRange(0, self.lenframes, padding=0)
        self.grid.addWidget(self.particles_vs_time_plot, 4, 0, 1, -1)

        self.modeSelector = RadioDemo(['TrackPy', 'SimpleTracking 1step', 'SimpleTracking 2step'], 0, self.update)
        self.mode = ''
        self.grid.addLayout(self.modeSelector.layout, 3, 0, 1, 2)
        self.displaySelector = RadioDemo(['Center', 'Area', 'None'], 1, self.update)
        self.grid.addLayout(self.displaySelector.layout, 3, 2, 1, -1)

        self.lowerStatusText = QLabel()
        self.grid.addWidget(self.lowerStatusText, 6, 0, -1, -1)

        self.imageViewBoxVerticalSliders = []

        self.imageViewBoxVerticalSliders.append(
            MySlider.DiameterVerticalSlider(
                3, 51, integer=True, top=False, title='Diameter'))
        self.imageViewBoxVerticalSliders[-1].slider.valueChanged.connect(
            lambda value: self.update('DiameterSlider', value)) # If several slots are connected to one signal, the slots will be executed one after the other, in the order they have been connected, when the signal is emitted.
        self.grid.addWidget(self.imageViewBoxVerticalSliders[-1], 1, 2)

        self.imageViewBoxVerticalSliders.append(
            MySlider.VerticalSlider(
                0, 50, integer=True, top=False, title='Maxsize'))
        self.imageViewBoxVerticalSliders[-1].slider.valueChanged.connect(
            lambda value: self.update('MaxsizeSlider', value)) # If several slots are connected to one signal, the slots will be executed one after the other, in the order they have been connected, when the signal is emitted.
        self.grid.addWidget(self.imageViewBoxVerticalSliders[-1], 1, 3)
        
        self.imageViewBoxVerticalSliders.append(
            MySlider.SeparationWidget(
                0, 50, integer=True, top=False, title='Separation')
            )
        self.imageViewBoxVerticalSliders[-1].checkbox.stateChanged.connect(
            lambda value: self.update('SeparationCheckbox', value))
        self.imageViewBoxVerticalSliders[-1].slider.valueChanged.connect(
            lambda value: self.update('SeparationSlider', value)) # If several slots are connected to one signal, the slots will be executed one after the other, in the order they have been connected, when the signal is emitted.
        self.grid.addWidget(self.imageViewBoxVerticalSliders[-1], 1, 4)

        # Right plot start
        self.rightPlotOuterLayout = QVBoxLayout()
        self.setup_right_plot()
        self.grid.addLayout(self.rightPlotOuterLayout, 1, 5, 1, 1)
        # end og Right plot

        self.setLayout(self.grid) 
        self.setWindowTitle('Particle Tracker based on TrackPy')
        self.setWindowIcon(QtGui.QIcon('Icon.PNG'))


        #self.update()
    def setup_right_plot(self):
        self.rightPlot = pg.PlotWidget(labels={'left':'Size', 'bottom':'Mass'})
        self.rightPlotItem = self.rightPlot.getPlotItem()
        self.rightPlotItemScat = pg.ScatterPlotItem()
        self.rightPlotItem.addItem(self.rightPlotItemScat)
        self.rightPlotOuterLayout.addWidget(self.rightPlot)
        self.rightPlotLowerLayout = QHBoxLayout()
        self.rightPlotOuterLayout.addLayout(self.rightPlotLowerLayout)
        self.xAxisLabel = QLabel('x Axis:')
        self.yAxisLabel = QLabel('y Axis:')
        self.xAxisSelector = QtGui.QComboBox()
        self.yAxisSelector = QtGui.QComboBox()
        self.xAxisSelector.addItems(['x', 'y', 'mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep', 'frame'])
        self.yAxisSelector.addItems(['x', 'y', 'mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep', 'frame'])
        self.xAxisSelector.setCurrentIndex(2)
        self.yAxisSelector.setCurrentIndex(3)
        self.xAxisSelector.currentTextChanged.connect(lambda value: self.update('xAxisSelector.currentTextChanged', value))
        self.yAxisSelector.currentTextChanged.connect(lambda value: self.update('yAxisSelector.currentTextChanged', value))
        self.rightPlotLowerLayout.addWidget(self.yAxisLabel)
        self.rightPlotLowerLayout.addWidget(self.yAxisSelector)
        self.rightPlotLowerLayout.addWidget(self.xAxisLabel)
        self.rightPlotLowerLayout.addWidget(self.xAxisSelector)

        self.plots = [] # trajectories plots

    def on_lin_region_change(self):
        minmass, maxmass = self.linRegion.getRegion()
        if abs(minmass*self.linRegionUpperLimit/255 - self.minmass)/self.minmass > 10**-2:
            self.particles_vs_time_plot_item_scat.clear()        
        self.minmass = minmass*self.linRegionUpperLimit/255
        self.maxmass = maxmass*self.linRegionUpperLimit/255
        #print("On lin region change", minmass, maxmass, self.linRegionUpperLimit, self.minmass, self.maxmass)
        self.update('on_lin_region_change')

    def get_all_sliders(self):
        if not self.running:
            #print("get_all_sliders")
            self.diameter = int(self.imageViewBoxVerticalSliders[0].getValue())
            #if self.diameter %2 == 0:
            #    self.diameter += 1
            self.maxsize = int(self.imageViewBoxVerticalSliders[1].getValue())
            self.separation = int(self.imageViewBoxVerticalSliders[2].getValue())
            self.invert = self.invertCheckbox.isChecked()
            return self.diameter, self.maxsize, self.separation
        else:
            self.set_all_sliders()
            
    def set_all_sliders(self):
        self.imageViewBoxVerticalSliders[0].setSliderPosition(self.diameter)
        self.imageViewBoxVerticalSliders[1].setSliderPosition(self.maxsize)
        self.imageViewBoxVerticalSliders[2].update_enable_state()
        self.imageViewBoxVerticalSliders[2].setSliderPosition(
            self.separation,
            self.diameter)

    def get_current_particles(self, mode):
        
        if mode[1] == 'TrackPy':
            return self.get_current_particles_trackpy()
        elif mode[1] == 'SimpleTracking 1step':
            return find_particles_first_frame(self.frames[self.frame_number],
                                              self.minmass,
                                              self.invert,
                                              min_size=self.diameter,
                                              max_size=self.maxsize,
                                              return_area=True)

    def get_current_particles_trackpy(self):
        maxsize=self.maxsize
        if self.imageViewBoxVerticalSliders[2].checkbox_enabled:
            separation = self.separation
        else:
            separation=None
        noise_size=1
        smoothing_size=None
        threshold=None
        invert=self.invert
        percentile=64
        topn=None
        preprocess=True
        max_iterations=10
        filter_before=None
        filter_after=None
        characterize=True
        engine='auto'
        output=None
        meta=None
        tmp = tp.locate(self.frames[self.frame_number], self.diameter, self.minmass,
                        maxsize, separation,
                              noise_size, smoothing_size, threshold, invert,
                              percentile, topn, preprocess, max_iterations,
                              filter_before, filter_after, characterize,
                              engine)
        self.particles = tmp
        #print(tmp)
        return tmp

    def update(self, *args, **kwargs):
        """
        Kwargs never happen.
        Args tell update what called it and the new value (of sliders...). Do not use this arg for different processing in update!
        Please, please. Dont.
        """
        try:
            self.plots
        except:
            # Setup not complete, abort.
            return
        
        if not self.running:
            print("Self.update", self.running, time(), *args, list(kwargs))
            t0 = time()
            self.clear_trajectories_plots()
            
            self.frame_number = self.framesSlider.slider.sliderPosition()
            self.get_all_sliders()
            self.set_all_sliders()

            #print("Start set image")
            self.imageImageItem.setImage(np.flip(self.frames[self.frame_number].T, axis=1))
            #print("End set image")

            mode = self.modeSelector.which()
            print('mode', mode)
            if mode != self.mode:
                self.mode = mode
                if mode[1] == 'TrackPy':
                    self.imageViewBoxVerticalSliders[0].slider.setMaximum(51)
                    self.imageViewBoxVerticalSliders[1].slider.setMaximum(50)
                    self.imageViewBoxVerticalSliders[0].title.setText('Diameter')
                else:
                    self.imageViewBoxVerticalSliders[0].slider.setMaximum(200)
                    self.imageViewBoxVerticalSliders[1].slider.setMaximum(5000)
                    self.imageViewBoxVerticalSliders[0].title.setText('Minsize')
            tmp = self.get_current_particles(mode)
            #print(tmp)
            if mode[1] == 'TrackPy':
                if len(tmp) >= 1:
                    tmpmaxmass = tmp.max()['mass']
                    if self.linRegionUpperLimit < 1.01*tmpmaxmass or self.linRegionUpperLimit > 1.10*tmpmaxmass:
                        if abs(self.maxmass - self.linRegionUpperLimit) < 10**-1:
                            self.maxmass = 1.05*tmpmaxmass
                        self.linRegionUpperLimit = 1.05*tmpmaxmass
                        
                    self.update_colorbar_texts_and_positions()
                    
                ccolors = convert_float_to_color(tmp.mass.tolist()/self.linRegionUpperLimit)
                display_mode = self.displaySelector.which()
                print('display_mode', display_mode)
                if display_mode[1] == 'Area':
                    self.imageScatter.setData(tmp.x, self.height - tmp.y, pxMode=False,
                        brush=ccolors, symbol='o', size=self.diameter)
                elif display_mode[1] == 'Center':
                    self.imageScatter.setData(tmp.x, self.height - tmp.y, pxMode=True,
                        brush=ccolors, symbol='+', size=20)
                else:
                    self.imageScatter.clear()
                cxdata = tmp[self.xAxisSelector.currentText()].tolist()
                cydata = tmp[self.yAxisSelector.currentText()].tolist()
                if 'y' == self.xAxisSelector.currentText():
                    cxdata = [self.height - aaa for aaa in cxdata]
                if 'y' == self.yAxisSelector.currentText():
                    cydata = [self.height - aaa for aaa in cydata]
                self.rightPlotItemScat.setData(cxdata, cydata, brush=ccolors)

                self.rightPlotItem.setLabel('bottom', self.xAxisSelector.currentText())
                self.rightPlotItem.setLabel('left', self.yAxisSelector.currentText())

                if args[0] not in ['framesSlider', 'on_lin_region_change',
                                   'xAxisSelector.currentTextChanged',
                                   'yAxisSelector.currentTextChanged']: # TODO. Please don't use this type of check.
                    # Does not work. Does not differentiate between user change and automatic colorbar change.
                    # TODO, fix this.
                    self.particles_vs_time_plot_item_scat.clear()
                    
                self.bottom_plot_update(len(tmp))
            else:
                self.linRegionUpperLimit = 256      
                self.update_colorbar_texts_and_positions()

                cx = np.array([k[0] for k in tmp[1]])
                cy = np.array([k[1] for k in tmp[1]])
                display_mode = self.displaySelector.which()
                print('display_mode', display_mode)
                if display_mode[1] == 'Area':
                    all_points = []
                    for k in tmp[1]:
                        all_points.extend(k[-1])
                    all_points = np.array(all_points)
                    if len(all_points) > 0:
                        cx = all_points.T[0]
                        cy = all_points.T[1]
                        self.imageScatter.setData(cy, self.height - cx, pxMode=False,
                            brush='b', symbol='s', size=1)
                    else:
                        self.imageScatter.clear()
                elif display_mode[1] == 'Center':
                    ccolors = 'b'
                    self.imageScatter.setData(cy, self.height - cx, pxMode=True,
                        brush=ccolors, symbol='+', size=20)
                else:
                    self.imageScatter.clear()
            
            self.lowerStatusText.setText("Updated in {:.3f}s".format(time()-t0))
        else:
            self.set_all_sliders()

    def update_from_calculation(self, frame, particles, image):
        self.clear_trajectories_plots()
        self.frame_number = frame
        diameter = self.diameter
        self.framesSlider.setSliderPosition(frame)
        self.set_all_sliders()
        
        #image = np.array(image)
        self.imageImageItem.setImage(np.flip(image.T, axis=1))
        tmp = particles

        mode = self.modeSelector.which()
        if mode[1] == 'TrackPy':
            if len(tmp) >= 1:
                tmpmaxmass = tmp.max()['mass']
                if self.linRegionUpperLimit < 1.01*tmpmaxmass:
                    if abs(self.maxmass - self.linRegionUpperLimit) < 10**-1:
                        self.maxmass = 1.05*tmpmaxmass
                    self.linRegionUpperLimit = 1.05*tmpmaxmass
                    
                self.update_colorbar_texts_and_positions()
                
            ccolors = convert_float_to_color(tmp.mass.tolist()/self.linRegionUpperLimit)
            
            self.imageScatter.setData(tmp.x, self.height - tmp.y, pxMode=False,
                brush=ccolors,
                size=diameter)
            cxdata = tmp[self.xAxisSelector.currentText()].tolist()
            cydata = tmp[self.yAxisSelector.currentText()].tolist()
            if 'y' == self.xAxisSelector.currentText():
                cxdata = [self.height - aaa for aaa in cxdata]
            if 'y' == self.yAxisSelector.currentText():
                cydata = [self.height - aaa for aaa in cydata]
            self.rightPlotItemScat.setData(cxdata, cydata, brush=ccolors)        

            self.rightPlotItem.setLabel('bottom', self.xAxisSelector.currentText())
            self.rightPlotItem.setLabel('left', self.yAxisSelector.currentText())
        else:
            #print(tmp)
            cx = np.array([k[0] for k in tmp])
            cy = np.array([k[1] for k in tmp])
            display_mode = self.displaySelector.which()
            #print('display_mode', display_mode)
            if display_mode[1] == 'Area':
                all_points = []
                for k in tmp:
                    all_points.extend(k[-1])
                all_points = np.array(all_points)
                if len(all_points) > 0:
                    cx = all_points.T[0]
                    cy = all_points.T[1]
                    self.imageScatter.setData(cy, self.height - cx, pxMode=False,
                        brush='b', symbol='s', size=1)
                else:
                    self.imageScatter.clear()
            elif display_mode[1] == 'Center':
                ccolors = 'b'
                self.imageScatter.setData(cy, self.height - cx, pxMode=True,
                    brush=ccolors, symbol='+', size=20)
            else:
                self.imageScatter.clear()            
        self.bottom_plot_update(len(particles))
        #print(len(particles))
        
    def update_colorbar_texts_and_positions(self, set_position=False):
        """
        Args:
         - tmpMaxMass: the maximal detected mass in an image.
         - minmass, maxmass: selected min and maxmass for filtering.
        """
        # update values
        if self.maxmass > self.linRegionUpperLimit: self.maxmass = self.linRegionUpperLimit
        self.colorbar_text[0].setText(str(int(self.linRegionUpperLimit)))
        self.colorbar_text[1].setText(str(int(self.maxmass)))
        self.colorbar_text[2].setText(str(int(self.minmass)))
        self.colorbar_text[1].setPos(0, self.maxmass/self.linRegionUpperLimit*255)
        self.colorbar_text[2].setPos(0, self.minmass/self.linRegionUpperLimit*255)
        if True:
            self.linRegion.setRegion((self.minmass/self.linRegionUpperLimit*255,
                                      self.maxmass/self.linRegionUpperLimit*255))

    def on_load_file(self):
        fileName, other = QFileDialog.getOpenFileNames(
            None, "Select file or files", "",
            "All Files (*);;Twv Files (*.twv);;PNG Files (*.png)")
        print("on_load_file", len(fileName), other)
        if len(fileName) == 0: return
        self.intype = fileName[0].split('.')[-1]
        self.filename = fileName[:]
        self.load_file(update=True)
            
    def start_processing(self):

        if (self.running):
            self.thread1.terminate()
            self.thread1.wait()
            #print("End of wait")
            self.sig.disconnect()
            del self.thread1
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.playButton.setText("Start")
            self.running = False
            self.lowerStatusText.setText("Processing thread terminated")
            
        else:
            try:
                if len(self.filename) > 1:
                    self.folder = self.filename[0].replace('\\', '/').split('/')
                    self.folder.pop(-1)
                    self.folder = '/'.join(self.folder) + '_checkpoints'
                else:
                    self.folder = self.filename[0][:-4] + '_checkpoints'
                os.mkdir(self.folder) # hardcoded extension length
                # not compatible with slicing of frames. It sucks. TODO
            except FileExistsError:
                pass
            except Exception:
                print("Re-raising exception")
                raise
            self.get_all_sliders()
            mode = self.modeSelector.which()
            if mode[1] == 'TrackPy':
                self.thread1 = TrackPyProcessThread(self.filename[0], self.frames, folder=self.folder,
                                         diameter=self.diameter, maxsize=self.maxsize,
                                         separation=self.separation, minmass=self.minmass,
                                         invert=self.invert)
            else:
                self.thread1 = SimpleTracking1StepProcessThread(
                    self.filename[0], self.frames, self.folder,
                    treshold=self.minmass,
                    invert=self.invert,
                    max_size=self.maxsize,
                    min_size=self.diameter)
            self.sig.connect(self.thread1.on_source)
            self.sig.emit('test')
            self.thread1.start()
            self.thread1.sig1.connect(self.self_receive_data)
            self.thread1.sig1a.connect(self.self_receive_data_a)
            self.thread1.sig2.connect(self.self_receive_text)
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
            self.playButton.setText("Stop")
            self.particles_vs_time_plot_item_scat.clear()
            self.running = True

    def self_receive_data(self, frame, fname, image):
        #print(frame, fname)
        tmp = pd.read_pickle(self.folder + '/{:}.pkl'.format(frame))
        self.update_from_calculation(frame, tmp, image)

    def self_receive_data_a(self, frame, particles, image):
        #print(frame, fname)
        self.update_from_calculation(frame, particles, image)

    def self_receive_text(self, text):
        self.lowerStatusText.setText(text)
        if 'Finished' in text:
            self.start_processing()
            self.display_trajectories()

    def tmp_to_float(self, tmp):
        for i in range(len(tmp)):
            try:
                tmp[i] = float(tmp[i])
            except:
                tmp[i] = float('NaN')
        return tmp

    def clear_trajectories_plots(self):
        for i in self.plots:
            i.clear()
        self.plots = []

    def display_trajectories(self):
        # Optional.
        self.xAxisSelector.setCurrentIndex(0)
        self.yAxisSelector.setCurrentIndex(1)
        # read data
        trajectories = []
        flag = False
        with open(self.filename[0][:-4] + '_out.dat', 'r') as f:
            for i in f:
                tmp = i.strip('\n').split('\t')
                tmp = self.tmp_to_float(tmp)
                if not flag:
                    flag = True
                    no_of_trajectories = len(tmp)//2-7
                    trajectories = [[] for i in range(no_of_trajectories)]
                for j in range(14, 14+2*no_of_trajectories, 2):
                    if (math.isnan(tmp[j]) and math.isnan(tmp[j+1])):
                        continue
                    trajectories[j//2-7].append([tmp[j], tmp[j+1]])
        trajectories = np.array(trajectories)
        self.clear_trajectories_plots()
        for i in range(no_of_trajectories):
            trajectories[i] = np.array(trajectories[i])
            self.plots.append(
                self.rightPlotItem.plot(trajectories[i].T[0], self.height - trajectories[i].T[1], pen=pg.mkPen(convert_rgb_to_hex(
                    viridis(i/no_of_trajectories)), width=5)))
        self.rightPlotItemScat.clear()
        print("Plotted")

    def bottom_plot_update(self, no_of_particles):
        # options
        # add scatter point
        # update line plots
        if self.running:
            pass
            #self.plots.append(
            #    self.rightPlotItem.plot(trajectories[i].T[0], trajectories[i].T[1], pen=pg.mkPen(convert_rgb_to_hex(
            #        viridis(i/no_of_trajectories)), width=5)))
            # TODO change to line.

            #self.particles_vs_time_plot_item_scat.addPoints([self.frame_number], [no_of_particles])
            # this freezes after a couple of thousand of points.
        else:
            self.particles_vs_time_plot_item_scat.addPoints([self.frame_number], [no_of_particles])

    def closeEvent(self, event):
        print("Closing", event)
        try:
            self.thread1.terminate()
            self.thread1.wait()
            self.sig.disconnect()
            del self.thread1
        except:
            pass
        print("Closed")

# TODO correct thread killing
# TODO Start/Stop at any step.
# TODO Default separation checkbox.
# TODO Cancel on file dialog.
# TODO improve everything.
# TODO Display number of particles/trajectories vs time/frame.
# TODO Switch between full/empty circle/center point display of points.
# Hmm, ransparent colors?
# TODO Crop for faster execution.
        
if __name__ == '__main__':

    filename = "tests/test_example.twv"

    sys._excepthook = sys.excepthook 
    def exception_hook(exctype, value, traceback):
        print(exctype, value, traceback)
        sys._excepthook(exctype, value, traceback) 
        sys.exit(1) 
    sys.excepthook = exception_hook 
    
    app = QApplication(sys.argv)
    #fileName, _ = QFileDialog.getOpenFileName(None,"QFileDialog.getOpenFileName()", "","Twv Files (*.twv);;All Files (*);;Python Files (*.py)")

    #frames = pims.open(filename)#[:50]
    
    #app = QApplication(sys.argv)
    ex = Example([filename])
    ex.show()
    ex.update('outside update call')
    ex.update('enother outside update call')
    sys.exit(app.exec_())
