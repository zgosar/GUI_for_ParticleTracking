import trackpy as tp
import warnings
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
from time import sleep
import os

import pims
import TWV_Reader


from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import pyqtSignal

class ProcessThread(QtCore.QThread):
    sig1 = pyqtSignal(int, str, np.ndarray)
    # int frame index, str filename, image
    sig1a = pyqtSignal(int, pd.core.frame.DataFrame, np.ndarray)
    # the same as sig1, except it doesn't save particles to
    # file and sends the filename, but sends the particles directly.
    sig2 = pyqtSignal(str)

    def __init__(self, filename, frames, folder, parent=None, diameter=11, minmass=100, maxsize=None, separation=None,
          noise_size=1, smoothing_size=None, threshold=None, invert=False,
          percentile=64, topn=None, preprocess=True, max_iterations=10,
          filter_before=None, filter_after=None, characterize=True,
          engine='auto', output=None, meta=None):
        QtCore.QThread.__init__(self, parent)
        self.filename = filename
        self.frames = frames
        self.folder = folder
        self.diameter = diameter
        self.minmass = minmass
        self.maxsize = maxsize
        print("Processing thread, maxsize", maxsize)
        self.separation = separation
        self.noise_size = noise_size
        self.smoothing_size = smoothing_size
        self.threshold = threshold
        self.invert = invert        
        self.percentile = percentile
        self.topn = topn
        self.preprocess =preprocess
        self.max_iterations = max_iterations
        self.filter_before = filter_before
        self.filter_after = filter_after
        self.characterize = characterize
        self.engine = engine
        self.output = output
        self.meta = meta

    def on_source(self, lineftxt):
        # this will be connected to some event/emmit.
        self.source_txt = lineftxt
        #print("Received", lineftxt)
        #print()

    def run_old(self):
        self.running = True
        while self.running:
            try:
                for i in range(0, 100):
                    self.sig1.emit(str(self.frames[i][0][0]), i, 100)
                    sleep(1.5)
            except Exception as err:
                self.sig1.emit(str(err), 0, 1)
                break

    def run(self, save_checkpoints=False):
        try:
            self.sig2.emit("Locating particles...")
            particles = self.batch_with_checkpointing(save_checkpoints=save_checkpoints)
            self.sig2.emit("Loading all particles to RAM...")
            if save_checkpoints:
                particles = self.load_particles()
            else:
                particles.reset_index(drop=True, inplace=True) # drop deletes the previous index.
                #inplace=True is the same as f = f.reset_index(inplace=False) except it may improve memory usage
                np.save(self.filename[:-4] + '_particles', particles)
            self.sig2.emit("Linking particles...")
            trajectories = self.link_particles()
            self.sig2.emit("Getting trap data...")
            print("get_all_tweezer_positions CALL")
            #self.frames.check_for_time_jumps()
            if self.filename[-4:] == '.twv':
                times, laserPowers, traps = self.frames.get_all_tweezer_positions()
            else:
                times = [i for i in range(len(self.frames))]
                laserPowers = [0 for i in times]
                traps = [[[0 for i in range(3)] for j in times] for k in range(4)]
            self.sig2.emit("Saving everything.")
            self.save_everything(trajectories, times, laserPowers, traps)
            self.sig2.emit("Finished.")
            # combine trajctories with trap data
            
        except Exception as err:
            print(str(err.__class__.__name__), err)
            import traceback
            traceback.print_exc()

    def save_everything(self, trajectories, times, laserPowers, traps):
        max_particles = int(round(trajectories.max()['particle']))
        with open(self.filename[:-4] + '_out.dat', 'w') as f:
            for i in range(len(self.frames)):
                self.sig2.emit("Saving everything. {:}/{:}".format(i, len(self.frames)))
                tmp = ''
                tmp += str(times[i]) + '\t'
                tmp += str(laserPowers[i]) + '\t'
                for j in range(4):
                    for k in range(3):
                        traps[j]
                        traps[j][i]
                        traps[j][i][k]
                        tmp += str(traps[j][i][k]) + '\t'
                for j in range(max_particles+1):
                    tmp_particle = trajectories.loc[
                        trajectories['particle'] == j].loc[
                            trajectories['frame'] == i]
                    #print(len(tmp_particle))
                    if len(tmp_particle) == 0:
                        tmp += '\t\t'
                    else:
                        tmp += str(tmp_particle.iloc[0]['y']) + '\t'
                        tmp += str(tmp_particle.iloc[0]['x']) + '\t'
                #print(tmp)
                tmp += '\n'
                f.write(tmp)
        print("Saved to", self.filename[:-4] + '_out.dat')
   
    def load_particles(self):
        f = pd.DataFrame()
        all_fs = []
        folder = self.folder
        for i in range(len(self.frames)):
            if i%1000 == 0: print(i)
            tmp = pd.read_pickle(self.folder + '/{:}.pkl'.format(i))
            tmp = tmp.drop(columns=['mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep'])
            # if resume from checkpoints is needed, then this need to be removed produce any better results the second time.
            all_fs.append(tmp)
        f = pd.concat(all_fs)
        f.reset_index(drop=True, inplace=True) # drop deletes the previous index.
        #inplace=True is the same as f = f.reset_index(inplace=False) except it may improve memory usage
        np.save(self.filename[:-4] + '_particles', f)
        return f

    def link_particles(self):
        f = np.load(self.filename[:-4] + '_particles.npy')
        if f.shape[1] == 3:
            f = pd.DataFrame(f, columns=['x', 'y', 'frame',])
        else:
            f = pd.DataFrame(f, columns=['x', 'y', 'mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep', 'frame'])
            f = f.drop(columns=['mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep'])
            # always drop? Saves memory (RAM) also down the line.
        t = tp.link_df(f, 15, memory=10)
        # TODO disable print
        # TODO display
        # TODO change parameters
        # TODO emit msgs.
        #print("Done link")
        np.save(self.filename[:-4] + '_trajectories', t)
        #print("Saved link")
        # TODO Filter
        # t1 = tp.filter_stubs(t, 50) # filter AFTER saving. So you can filter differently next time
      
        return t

    def batch_with_checkpointing(self, save_checkpoints=True):
        """ Based on batch from TrackPy. Simplified to use in the GUI.
        
        Locate Gaussian-like blobs of some approximate size in a set of images.

        Preprocess the image by performing a band pass and a threshold.
        Locate all peaks of brightness, characterize the neighborhoods of the peaks
        and take only those with given total brightness ("mass"). Finally,
        refine the positions of each peak.

        Parameters
        ----------
        frames : list (or iterable) of images
        diameter : odd integer or tuple of odd integers
            This may be a single number or a tuple giving the feature's
            extent in each dimension, useful when the dimensions do not have
            equal resolution (e.g. confocal microscopy). The tuple order is the
            same as the image shape, conventionally (z, y, x) or (y, x). The
            number(s) must be odd integers. When in doubt, round up.
        minmass : float
            The minimum integrated brightness.
            Default is 100 for integer images and 1 for float images, but a good
            value is often much higher. This is a crucial parameter for eliminating
            spurious features.
            .. warning:: The mass value was changed since v0.3.3
        maxsize : float
            maximum radius-of-gyration of brightness, default None
        separation : float or tuple
            Minimum separtion between features.
            Default is diameter + 1. May be a tuple, see diameter for details.
        noise_size : float or tuple
            Width of Gaussian blurring kernel, in pixels
            Default is 1. May be a tuple, see diameter for details.
        smoothing_size : float or tuple
            The size of the sides of the square kernel used in boxcar (rolling
            average) smoothing, in pixels
            Default is diameter. May be a tuple, making the kernel rectangular.
        threshold : float
            Clip bandpass result below this value.
            Default, None, defers to default settings of the bandpass function.
        invert : boolean
            Set to True if features are darker than background. False by default.
        percentile : float
            Features must have a peak brighter than pixels in this
            percentile. This helps eliminate spurious peaks.
        topn : integer
            Return only the N brightest features above minmass.
            If None (default), return all features above minmass.
        preprocess : boolean
            Set to False to turn off bandpass preprocessing.
        max_iterations : integer
            max number of loops to refine the center of mass, default 10
        filter_before : boolean
            filter_before is no longer supported as it does not improve performance.
        filter_after : boolean
            This parameter has been deprecated: use minmass and maxsize.
        characterize : boolean
            Compute "extras": eccentricity, signal, ep. True by default.
        engine : {'auto', 'python', 'numba'}
        output : {None, trackpy.PandasHDFStore, SomeCustomClass}
            If None, return all results as one big DataFrame. Otherwise, pass
            results from each frame, one at a time, to the put() method
            of whatever class is specified here.
        meta : filepath or file object, optional
            If specified, information relevant to reproducing this batch is saved
            as a YAML file, a plain-text machine- and human-readable format.
            By default, this is None, and no file is saved.

        Returns
        -------
        DataFrame([x, y, mass, size, ecc, signal])
            where mass means total integrated brightness of the blob,
            size means the radius of gyration of its Gaussian-like profile,
            and ecc is its eccentricity (0 is circular).

        See Also
        --------
        locate : performs location on a single image
        minmass_v03_change : to convert minmass from v0.2.4 to v0.3.0
        minmass_v04_change : to convert minmass from v0.3.3 to v0.4.0

        Notes
        -----
        This is an implementation of the Crocker-Grier centroid-finding algorithm.
        [1]_

        Locate works with a coordinate system that has its origin at the center of
        pixel (0, 0). In almost all cases this will be the topleft pixel: the
        y-axis is pointing downwards.

        References
        ----------
        .. [1] Crocker, J.C., Grier, D.G. http://dx.doi.org/10.1006/jcis.1996.0217

        """
        frames = self.frames
        diameter = self.diameter
        minmass = self.minmass
        maxsize = self.maxsize
        separation = self.separation
        noise_size = self.noise_size
        smoothing_size = self.smoothing_size
        threshold = self.threshold
        invert = self.invert        
        percentile = self.percentile
        topn = self.topn
        preprocess = self.preprocess
        max_iterations = self.max_iterations
        filter_before = self.filter_before
        filter_after = self.filter_after
        characterize = self.characterize
        engine = self.engine
        output = self.output
        meta = self.meta
        folder = self.folder
        
        # Gather meta information and save as YAML in current directory.
        timestamp = pd.datetime.utcnow().strftime('%Y-%m-%d-%H%M%S')
        try:
            source = frames.filename
        except:
            source = None
        """
        meta_info = dict(timestamp=timestamp,
                         trackpy_version=tp.__version__,
                         source=source, diameter=diameter, minmass=minmass,
                         maxsize=maxsize, separation=separation,
                         noise_size=noise_size, smoothing_size=smoothing_size,
                         invert=invert, percentile=percentile, topn=topn,
                         preprocess=preprocess, max_iterations=max_iterations)

        if meta:
            if isinstance(meta, six.string_types):
                with open(meta, 'w') as file_obj:
                    record_meta(meta_info, file_obj)
            else:
                # Interpret meta to be a file handle.
                record_meta(meta_info, meta)
        """
        
        all_features = []
        for i, image in enumerate(frames):
            if hasattr(image, 'frame_no') and image.frame_no is not None:
                frame_no = image.frame_no
                # If this works, locate created a 'frame' column.
            else:
                frame_no = i

            features = tp.locate(image, diameter, minmass, maxsize, separation,
                              noise_size, smoothing_size, threshold, invert,
                              percentile, topn, preprocess, max_iterations,
                              filter_before, filter_after, characterize,
                              engine)
            #print(features)

            features['frame'] = i  # just counting iterations # ERROR this was wrongly indented
            # this was in else that is now above.

            #print("Frame {:}/{:}: {:} features".format(frame_no, len(frames), len(features)))
            if len(features) == 0:
                continue
            
            if output is None:
                if save_checkpoints:
                    features.to_pickle(folder + '/{:}.pkl'.format(frame_no))
                    self.sig1.emit(frame_no, folder + '/{:}.pkl'.format(frame_no),
                               image)
                else:
                    self.sig1a.emit(frame_no, features,
                               image)
                self.sig2.emit("Locating particles... {:} particles on frame {:}".format(len(features), frame_no))
                # TODO: Do not append features to all_features if there is save checkpoints happening.
                all_features.append(features)
            else:
                output.put(features)
            
        if output is None:
            if len(all_features) > 0:
                return pd.concat(all_features).reset_index(drop=True)
            else:  # return empty DataFrame
                warnings.warn("No maxima found in any frame.")
                return pd.DataFrame(columns=list(features.columns) + ['frame'])
        else:
            return output
