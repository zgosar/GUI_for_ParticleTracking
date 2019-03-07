import trackpy as tp
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import time
import os

import pims
import TWV_Reader


def save_everything(filename, frames, trajectories, times, laserPowers, traps):
    max_particles = int(round(trajectories.max()['particle']))
    with open(filename[:-4] + '_out.dat', 'w') as f:
        for i in range(len(frames)):
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
                    tmp += str(tmp_particle.iloc[0]['x']) + '\t'
                    tmp += str(tmp_particle.iloc[0]['y']) + '\t'
            #print(tmp)
            tmp += '\n'
            f.write(tmp)

if __name__ == '__main__':
    
    filename = "passiveInTrapP1.twv"

    frames = pims.open(filename)
    frames.set_end_frame(50)

    metadata = frames.get_all_metadata()
    print(metadata)
    times, laserPowers, traps = frames.get_all_tweezer_positions()

    f = tp.batch(frames, 25, minmass=1000, invert=False)
    # diameter is 25. Must be odd number.
    # Verjetno lahko tudi z GUI določiš zgornja parametra, diameter in minmass,
    # potem pa na roke poženeš tole. 

    t = tp.link_df(f, 15, memory=10)
    # ta dva parametra spreminjaj po potrebi. Glej trackpy dokumentacijo.

    save_everything(filename, frames, t, times, laserPowers, traps)
