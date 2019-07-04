import trackpy as tp
import warnings
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
from time import sleep
import os
import queue

import pims
import TWV_Reader


from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import pyqtSignal

class TrackPyProcessThread(QtCore.QThread):
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


class SimpleTracking1StepProcessThread(QtCore.QThread):
    sig1 = pyqtSignal(int, str, np.ndarray)
    # int frame index, str filename, image
    sig1a = pyqtSignal(int, list, np.ndarray)
    # the same as sig1, except it doesn't save particles to
    # file and sends the filename, but sends the particles directly.
    sig2 = pyqtSignal(str)

    def __init__(self, filename, frames, folder, parent=None,
                 treshold=100, invert=False, min_size=8,
                 max_size=250000, max_distance=50):
        QtCore.QThread.__init__(self, parent)
        self.filename = filename
        self.frames = frames
        self.folder = folder
        self.treshold = treshold
        self.invert = invert
        self.min_size = min_size
        self.max_size = max_size
        self.max_distance = max_distance

    def on_source(self, lineftxt):
        # this will be connected to some event/emmit.
        self.source_txt = lineftxt
        #print("Received", lineftxt)
        #print()

    def run(self, save_checkpoints=False):
        try:
            self.sig2.emit("Locating particles...")
            particles = self.simple_tracking()
            self.sig2.emit("Getting trap data...")
            print("get_all_tweezer_positions CALL")
            #self.frames.check_for_time_jumps()
            if self.filename[-4:] == '.twv':
                times, laser_powers, traps = self.frames.get_all_tweezer_positions()
            else:
                times = [i for i in range(len(self.frames))]
                laser_powers = [0 for i in times]
                traps = [[[0 for i in range(3)] for j in times] for k in range(4)]
            self.sig2.emit("Saving everything.")
            self.save_tracked_data(
                self.filename[:-4] + '_out.dat',
                len(self.frames),
                particles, times, laser_powers, traps)
            self.sig2.emit("Finished.")
            # combine trajctories with trap data
            
        except Exception as err:
            print(str(err.__class__.__name__), err)
            import traceback
            traceback.print_exc()

    def flood_fill(self, frame, flood_frame, start_x, start_y, particle_number, treshold=100, invert=False, return_area=False):
        """
        An implementation of a common flood fill algorithm.
        Pixels with brightness above treshold are considered to be inside (below treshold if invert is True).
        flood_frame and particle_number are used to remember where particles were.
        Start position is start_x, start_y, where it is assumed to be above treshold and not checked.

        It also calculates on the fly the center of the pixel.

        Args:
         - frame: image array
         - flood_frame: Same shape as frame. 0 where there were no particles found yet.
         - prev_x, prev_y: Previous positions of particle.
         - particle_number: Used to mark this particle in flood_frame
         - treshold, invert, min_size, max_size, max_distance: See simple_tracking.
         - return_area: 

        Returns:
         - flood_frame: Updated flood_frame.
         - particle: Array with data about particle. Position x, y, size in pixels, average brightness and normalization weight.
           the last element is optional list of coordinates where particle was found
        """
        q = queue.Queue()
        visited = set()
        q.put((start_x, start_y))
        if return_area:
            particle = [0, 0, 0, 0, 0, []]
        else:
            particle = [0, 0, 0, 0, 0]
        while not q.empty():
            cx, cy = q.get()
            if (cx, cy) in visited:
                continue
            if (cx >= len(frame) or cy >= len(frame[0])
                or cx < 0 or cy < 0):
                continue
            visited.add((cx, cy))
            if ((not invert and frame[cx][cy] > treshold) or
                (invert and frame[cx][cy] < treshold)):
                flood_frame[cx][cy] = particle_number
                particle[0] += cx*(frame[cx][cy] - treshold)
                particle[1] += cy*(frame[cx][cy] - treshold)
                particle[2] += 1
                particle[3] += frame[cx][cy]
                particle[4] += frame[cx][cy] - treshold
                if return_area:
                    particle[5].append([cx, cy])
                q.put((cx+1, cy))
                q.put((cx-1, cy))
                q.put((cx, cy+1))
                q.put((cx, cy-1))
        particle[0] /= particle[4]
        particle[1] /= particle[4]
        particle[3] /= particle[2]
        return flood_frame, particle
                

    def find_particles_first_frame(self, frame, treshold=100, invert=False, min_size=16, max_size=2500, return_area=False):
        """
        Finds particles on the first frame.

        Every pixel with brightness larger than treshold starts a flood fill around its position.
        Particles that are too small or too big (min_size and max_size are in total number of pixels)
        are filtered out.

        Args:
         - frame: image array
         - treshold, invert, min_size, max_size: See simple_tracking.

        Returns:
         - flood_frame: Array with the same shape as frame. With 0 where there were no particles found yet and
                        markers of particles where particles were found.
         - particle: Array with data about particle. Position x, y, size in pixels, average brightness and normalization weight.    
        """
        flood_frame = np.zeros_like(frame)
        particle_number = 1
        particles = []
        for cx in range(len(frame)):
            for cy in range(len(frame[0])):
                if (flood_frame[cx][cy] == 0 and
                    ((not invert and frame[cx][cy] > treshold) or
                     (invert and frame[cx][cy] < treshold))):
                    #print('start', cx, cy, particle_number)
                    flood_frame, particle = self.flood_fill(frame, flood_frame,
                                             cx, cy, particle_number, treshold=treshold,
                                             invert=invert, return_area=return_area)
                    if min_size < particle[2] < max_size:
                        particles.append(particle[:])
                    else:
                        pass
                        #print("filtering", particle)
                    particle_number += 1
        return flood_frame, particles

    def spiral(self, R):
        """
        Generates a square spiral walk around (0, 0).
        Adapted from https://stackoverflow.com/a/398302/
        """
        x = y = 0
        dx = 0
        dy = -1
        for i in range((2*R)**2):
            if (-R < x <= R) and (-R < y <= R):
                yield (x, y)
            if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
                dx, dy = -dy, dx
            x, y = x+dx, y+dy

    def find_particle_around_position(self, frame, flood_frame, prev_x, prev_y, particle_number,
                                      treshold=100, invert=False, min_size=16, max_size=2500, max_distance=50,
                                      return_area=False):
        """
        Finds a particle around prev_x, prev_y up to max_distance away (square with a 2*max_distance side).

        Args:
         - frame: image array
         - flood_frame: Same shape as frame. 0 where there were no particles found yet.
         - prev_x, prev_y: Previous positions of particle.
         - particle_number: Used to mark this particle in flood_frame
         - treshold, invert, min_size, max_size, max_distance: See simple_tracking.

        Returns:
         - flood_frame: Updated flood_frame.
         - particle: Array with data about particle. Position x, y, size in pixels, average brightness and normalization weight.
        """
        for dx, dy in self.spiral(max_distance):
            cx = prev_x + dx
            cy = prev_y + dy
            if (cx < len(flood_frame) and cy < len(flood_frame[cx]) and
                cx >= 0 and cy >= 0 and
                flood_frame[cx][cy] == 0 and
                frame[cx][cy] > treshold):
                flood_frame, particle = self.flood_fill(frame, flood_frame,
                                             cx, cy, particle_number, treshold=treshold,
                                             invert=invert, return_area=return_area)
                if min_size < particle[2] < max_size:
                    return flood_frame, particle[:]
        return flood_frame, [0, 0, 0, 0, 0, []]
        
    def simple_tracking(self):
        """
        Tracks particles on all frames using a simple flood-fill of pixels above treshold.
        Args:
         - frames: an array of frames. Each frame must be like 2D array with single numbers as values.
                   works with pims, but it is not needed.
         - treshold: Pixels above treshold are considered particle.
         - invert: If True, it looks for dark patches instead of bright.
         - min_size, max_size: If particle is below/above this pixel count, it is discarded.
         - max_distance: The distance in pixels, when we stop looking for a particle on the nex frame
                         (distance from the previous frame position). Distance is calculated as the
                         supremum distance (max of distances in x and y directions).

        Returns:
         - positions: Array. positions[frame number][particle number][x, y, other particle data]

        Note:
         - The new particle is searched for from the previous particles position in a spiral. That may
           introduce some bias when multiple particles are near eachother and jumps from one to the other.

        """
        frames = self.frames
        treshold = self.treshold
        invert = self.invert
        min_size = self.min_size
        max_size = self.max_size
        max_distance = self.max_distance
        return_area = True
        
        positions = [] # positions[frame][particle][x, y, ...]
        flood_frame, particles = self.find_particles_first_frame(frames[0],
            treshold=treshold, invert=invert, min_size=min_size, max_size=max_size,
            return_area=return_area)
        positions.append(particles[:])
        for i in range(1, len(frames)):
            flood_frame = np.zeros_like(frames[i])
            cparticles = []
            for particle_number in range(len(positions[0])):
                flood_frame, particle = self.find_particle_around_position(frames[i], flood_frame,
                    int(round(positions[-1][particle_number][0])),
                    int(round(positions[-1][particle_number][1])),
                    particle_number + 1, treshold=treshold, invert=invert, # particle_number + 1 to avoid using 0.
                    min_size=min_size, max_size=max_size,
                    max_distance=max_distance, return_area=return_area)
                cparticles.append(particle)
            positions.append(cparticles)
            #cparticles = np.array(cparticles)
            self.sig1a.emit(i, cparticles, self.frames[i])
        return positions
        

    def save_tracked_data(self, filename, Nframes, trajectories, times, laser_powers, traps):
        """
        Converts all data to a (.dat) file.
        Important NOTE:
        The format will change in the future. Particle data and trap data will go into separate files,
        and additional trap metadata will be saved.
        
        Inputs:
         - filename: Output filename.
         - Nframes: Number of frames
         - trajectories: trajectories - different order than GUIless version
         - times: List of frame times.
         - laser powers: List of laser powers for every frame.
         - traps: List of traps data for every frame. traps[trap number][frame number][position_x/position_y/power]
         
        Output format, tab separated:
         - time, laser power, trap_1_power, trap_1_x, trap_1, y,
             the same for traps 2-4, particle_1_x, particle_1_y,
             the same for all particles
        If a particle is missing from a frame, empty string ('') is placed
        instead of coordinates.
        """
        max_particles = len(trajectories[0])
        with open(filename, 'w') as f:
            for i in range(Nframes):
                tmp = ''
                tmp += str(times[i]) + '\t'
                tmp += str(laser_powers[i]) + '\t'
                for j in range(4): # for j in traps
                    for k in range(3): # for k in x/y/power of a trap
                        tmp += str(traps[j][i][k]) + '\t'
                for j in range(max_particles):
                    #print(1, j, i, '|', len(trajectories), len(trajectories[0]), len(trajectories[0][0]))
                    tmp += str(trajectories[i][j][0]) + '\t'
                    tmp += str(trajectories[i][j][1]) + '\t'
                tmp += '\n'
                f.write(tmp)
