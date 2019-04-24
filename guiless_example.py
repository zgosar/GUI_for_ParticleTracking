import pims
import trackpy as tp
import TWV_Reader

def save_everything(filename, frames, trajectories, times, laser_powers, traps):
    """
    Converts all data to .dat file.
    Important NOTE:
    The format will change in the future. Particle data and trap data will go into separate files,
    and additional trap metadata will be saved.
    Current format:
     - time, laser power, trap_1_power, trap_1_x, trap_1, y,
     the same for traps 2-4, particle_1_x, particle_1_y,
     the same for all particles
    if a particle is missing from a frame, empty string '' is placed
    instead of coordinates. 
    """
    max_particles = int(round(trajectories.max()['particle']))
    with open(filename, 'w') as f:
        for i in range(len(frames)):
            tmp = ''
            tmp += str(times[i]) + '\t'
            tmp += str(laser_powers[i]) + '\t'
            for j in range(4):
                for k in range(3):
                    tmp += str(traps[j][i][k]) + '\t'
            for j in range(max_particles+1):
                tmp_particle = trajectories.loc[
                    trajectories['particle'] == j].loc[
                        trajectories['frame'] == i]
                if tmp_particle.empty:
                    tmp += '\t\t'
                else:
                    tmp += str(tmp_particle.iloc[0]['x']) + '\t'
                    tmp += str(tmp_particle.iloc[0]['y']) + '\t'
            tmp += '\n'
            f.write(tmp)

if __name__ == '__main__':

    FILENAME = 'passiveInTrapP1.twv'

    frames = pims.open(FILENAME)
    frames.set_end_frame(50)

    metadata = frames.get_all_metadata()

    times, laser_powers, traps = frames.get_all_tweezer_positions()

    f = tp.batch(frames, 25, minmass=1000, invert=False)
    # Diameter is 25. Must be odd number.
    # The parameters for batch are recommended to be determined using GUI.

    t = tp.link_df(f, 15, memory=10)
    # See trackpy documentation for above parameters.

    save_everything(FILENAME[:-4] + '_out.dat', frames, t, times, laser_powers, traps)
