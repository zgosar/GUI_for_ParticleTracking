"""
Defines the TWV_Reader for pims.
The reader reads .twv files from Optical Tweezers.
Requirements ctypes, pims, numpy
Not mandatory requirements:
-matplotlib
"""


from pims import FramesSequence, Frame
from ctypes import * # todo remove * import.
import numpy as np
import struct

verbose_level = 2

class TArFrameROI(Structure):
    _pack_ = 1
    _fields_ = [("Left", c_uint16),
                ("Top", c_uint16),
                ("Width", c_uint16),
                ("Height", c_uint16)
                ]

class TArFrameData(Structure):
    _pack_ = 1
    _fields_ = [("HeaderSize", c_uint16),
                ("FrameDataIncl", c_uint16),
                ("ROI", TArFrameROI),
                ("BytesPerPixel", c_uint16),
                ("FrameRate", c_double),
                ("Exposure", c_double),
                ("Gain", c_double),
                ("RecTrapDataNo", c_uint16),
                ]

class TArCalibrationData(Structure):
    _pack_ = 1
    _fields_ = [("ImageToSampleScale", c_float)               
                ]

class TArVideoHeader(Structure):
    _pack_ = 1
    _fields_ = [("VideoID", c_uint16),
                ("VideoVersion", c_uint16),
                ("VideoLicence", c_uint16),
                ("VideoHeaderSize", c_uint32),
                ("RecordedFrames", c_uint32),
                ("FrameData", TArFrameData),
                ("CalibrationData", TArCalibrationData)
                ]

class TArTrapData(Structure):
    _pack_ = 1
    _fields_ = [("PositionX", c_float),
                ("PositionY", c_float),
                ("Intensity", c_float)
                ]

class TArFrameHeader(Structure):
    _pack_ = 1
    _fields_ = [("FrameNumber", c_uint32),
                ("FrameTime", c_float),
                ("LaserPower", c_float),
                ("CalibrationData1", TArTrapData),
                ("CalibrationData2", TArTrapData),
                ("CalibrationData3", TArTrapData),
                ("CalibrationData4", TArTrapData)                
                ]

def display_attr(obj):
    """
    Prints obj.
    """
    for field_name, field_type in obj._fields_:
        print(field_name, getattr(obj, field_name))

def test():
    "Obsolete"

    f.seek(getattr(videoHeader, "VideoHeaderSize"))
    f.readinto(frameHeader)
    
    frameData=getattr(videoHeader, "FrameData")
    display_attr(frameData)
    frameROI=getattr(frameData, "ROI")
    display_attr(frameROI)
    
    print(frameROI)
    imageW =getattr(frameROI, "Width")
    imageH =getattr(frameROI, "Height")
    imageSize= imageW*imageH
    print( imageW, imageH,imageSize)
    
    
    print("Frame header: ====================")
    display_attr(frameHeader)
    f.seek(imageSize,1)
    
    f.readinto(frameHeader)
    print("NEW Frame header: ====================")
    for field_name, field_type in frameHeader._fields_:
        print(field_name, getattr(frameHeader, field_name))
    f.close()



class TWV_Reader(FramesSequence):
    #propagate_attrs = 'filename'
    
    def __init__(self, filename):
        self.filename = filename
        self.f = open(filename, "rb")
        
        self.videoHeader = TArVideoHeader()
        self.f.readinto(self.videoHeader)
        self.f.seek(getattr(self.videoHeader, "VideoHeaderSize"))
        if self.videoHeader.FrameData.BytesPerPixel != 1:
            from warnings import warn
            warn("videoHeader.FrameData.BytesPerPixel is not 1")

        #metadata['frameHeader'] = TArFrameHeader()
        #f.readinto(metadata['frameHeader'])
        
        self._len =  self.videoHeader.RecordedFrames
        self._dtype =  np.uint8
        self._frame_shape = (self.videoHeader.FrameData.ROI.Width,
                             self.videoHeader.FrameData.ROI.Height)
        self.frame_header_size = self.videoHeader.FrameData.HeaderSize
        self.frame_size_bytes = self._frame_shape[0] * self._frame_shape[1] + self.frame_header_size

    def get_all_metadata(self):
        if 0:
            display_attr(self.videoHeader)
            print('---')
            display_attr(self.videoHeader.FrameData)
            print('---')
            display_attr(self.videoHeader.FrameData.ROI)
            print('---')
            display_attr(self.videoHeader.CalibrationData)

        out_dict = dict()
        out_dict['ROI'] = dict()
        out_dict['ROI']['Left'] = self.videoHeader.FrameData.ROI.Left
        out_dict['ROI']['Top'] = self.videoHeader.FrameData.ROI.Top
        out_dict['ROI']['Width'] = self.videoHeader.FrameData.ROI.Width
        out_dict['ROI']['Height'] = self.videoHeader.FrameData.ROI.Height
        out_dict['ImageToSampleScale'] = self.videoHeader.CalibrationData.ImageToSampleScale
        out_dict['FrameRate'] = self.videoHeader.FrameData.FrameRate
        out_dict['Exposure'] = self.videoHeader.FrameData.Exposure
        out_dict['Gain'] = self.videoHeader.FrameData.Gain
        return out_dict

    def set_end_frame(self, end_frame):
        self._len = end_frame

    def get_frame(self, frame_no):
        """
        Returns a frame (image).
        """
        self.f.seek(self.videoHeader.VideoHeaderSize) # absolute seek
        self.f.seek(frame_no * self.frame_size_bytes + self.frame_header_size, 1) # relative seek
        image = []
        unpack_format = '{:}B'.format(self._frame_shape[0])
        for i in range(self._frame_shape[1]):
            image.append(struct.unpack(
                unpack_format,
                self.f.read(self._frame_shape[0])))
        image = np.array(image, dtype = self._dtype)

        return Frame(image, frame_no=frame_no)

    def get_frame_time_old(self, frame_no):
        """
        OBSOLETE.
        Use get_frame_time.
        """
        frame_header_size = self.videoHeader.FrameData.HeaderSize
        self.f.seek(self.videoHeader.VideoHeaderSize)
        self.f.seek(frame_no * (self._frame_shape[0] * self._frame_shape[1] + frame_header_size),
                    1)
        frameHeader = TArFrameHeader()
        self.f.readinto(frameHeader)
        return frameHeader.FrameTime

    def get_frame_time(self, frame_no):
        self.f.seek(self.videoHeader.VideoHeaderSize) # absolute seek
        self.f.seek(frame_no * self.frame_size_bytes, 1) # relative seek
        frameHeader = TArFrameHeader()
        self.f.readinto(frameHeader)
        return frameHeader.FrameTime

    def __len__(self):
        return self._len

    @property
    def frame_shape(self):
        return self._frame_shape

    @property
    def pixel_type(self):
        return self._dtype

    @classmethod
    def class_exts(cls):
        return {'twv'} | super(TWV_Reader, cls).class_exts()

    def check_for_time_jumps(self, treshold=10**-2, show=False):
        """
        Checks if there are any unusual time jumps in the file.
        Returns True if data is OK (max_time-min_time)/min_time <= treshold.
        Kwargs:
         - treshold=10**-2: If relative error is below this treshold,
             video is considered OK.
         - show=False: If true, the time vs. frame number and time step are plotted.

        TODO: Compare to reported FPS, instead of with each other.
        """
        times = []
        self.f.seek(self.videoHeader.VideoHeaderSize) # absolute seek
        for i in range(self._len):
            frameHeader = TArFrameHeader()
            self.f.readinto(frameHeader)
            times.append(frameHeader.FrameTime)
            self.f.seek(self._frame_shape[0] * self._frame_shape[1], 1) # relative seek
            
        times = np.array(times)
        time_deltas = times[1:]-times[:-1]
        min_time = min(time_deltas)
        max_time = max(time_deltas)
        if verbose_level >= 2:
            print("Min frame time {:}, max frame time {:}".format(
                min_time, max_time))
            print(times)
        ret = True
        if (max_time-min_time)/min_time > treshold:
            if verbose_level >= 1:
                print("There are time jumps: min_time {:}, max_time {:},"
                  "relative difference {:}, treshold {:}".format(
                      min_time, max_time, (max_time-min_time)/min_time, treshold))
            ret = False
        if show:
            import matplotlib.pyplot as plt
            f, ax = plt.subplots(2, 1, sharex=True)
            ax[0].plot(range(len(times)), times)
            ax[0].set_title("Time vs frame number")
            
            ax[1].plot(range(len(times)-1), time_deltas)
            #ax[1].semilogy()
            ax[1].set_title("Time step between frames (log scale) vs frame number.\n"
                            "avg FPS = {:}".format(
                len(times)/(times[-1] - times[0])))
            plt.show()
            plt.cla()
            plt.clf()
            
        return ret

    def get_all_tweezer_positions(self, which=[0,1,2,3], fname=None):
        """
        TODO
        Returns all tweezers positions
        - [times, laserPower, (trapX, trapY, intensity) for each trap]
        Optional input: which:, specifies which tweezers positions to return.
        If fname is set, it is written to it.
        Otherwise it is returned as an array in the shape 
        """
        print("get_all_tweezer_positions")
        verbose_lvl = 0 # TODO, get from config.
        laserPowers = [] # TODO
        times = []
        traps = [[] for i in range(4)]
        self.f.seek(self.videoHeader.VideoHeaderSize) # absolute seek
        for i in range(len(self)):
            frameHeader = TArFrameHeader()
            self.f.readinto(frameHeader)
            times.append(frameHeader.FrameTime)
            laserPowers.append(frameHeader.LaserPower)
            for j in which:
                tmp = frameHeader.__getattribute__("CalibrationData{:}".format(j+1))
                traps[j].append([
                    tmp.PositionX,
                    tmp.PositionY,
                    tmp.Intensity])
            self.f.seek(self._frame_shape[0] * self._frame_shape[1], 1) # relative seek
            
        times = np.array(times)
        if fname is None:
            return times, laserPowers, traps
            # times, laserPowers, traps[which_trap][time][x, y, power]
        else:
            pass
            # TODO
            # this will not be used. This will be used together
            # with particle tracker.
            


if __name__ == '__main__':
    filename = "passiveInTrapP1.twv"
    filename = "F:/Gersak/190226_Voda23_2TrapsAS_20kHz.twv"
    #print(get_all_frame_times(filename, show=True, treshold=10**-2))
    a = TWV_Reader(filename)
    print('filename', a.filename)
    a.set_end_frame(50)
    c = a.get_all_metadata()
    #a.check_for_time_jumps(show=False)
    #times, laserPowers, traps = a.get_all_tweezer_positions()
    import cv2
    #cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    #for i in range(a.videoHeader.RecordedFrames):
    #    cv2.imshow('image', a[i])
    #    cv2.waitKey(1)
