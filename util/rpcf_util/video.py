# generate a video file from given data.

import os
import sys

import numpy as np

from rpcf.simulation import SimulationResults
import pylab

def make_pngs(casedir):
    # make sure directories exist
    if not os.path.isdir(os.path.join(os.getcwd(), 'videos/', os.path.basename(casedir[:-1]))):
        os.mkdir(os.path.join(os.getcwd(), 'videos/', os.path.basename(casedir[:-1])))

    # unpack simulation data
    sim = SimulationResults(casedir)

    # define contour levels
    levels = np.linspace(-10, 10, 100)
    for i in range(len(sim.t)):
        snap = sim[sim.t[i]]
        pylab.clf()
        pylab.contourf(snap.omega.z, snap.omega.y, snap.omega.data, levels, extend="both", cmap=cm.seismic)
        # pylab.contourf(snap.omega.z, snap.omega.y, snap.omega.data, extend="both", cmap=pylab.cm.seismic)
        pylab.gca().set_aspect(1)
        pylab.xticks([0, 2, 4, 6, 8])
        pylab.yticks([-1, 0, 1])
        pylab.xlabel('$z$')
        pylab.ylabel('$y$')
        pylab.gca().tick_params(axis='x', direction='out')
        pylab.gca().tick_params(axis='y', direction='out')
        pylab.savefig(os.path.join(os.getcwd(), 'videos/', os.path.basename(casedir[:-1]), 'frame%06d.png' % i), dpi=1000, bbox_inches='tight')
        print(i), len(sim.t)

def pngs2mp4(casedir, video_name, frame_rate=60):
    video_dir = os.path.join(os.getcwd(), 'videos/', os.path.basename(casedir[:-1]))
    ffmpeg_call = f"ffmpeg -r {frame_rate} -pattern_type glob -i '{video_dir}/frame*.png' -vcodec libx264 {video_dir}/{video_name}"
    os.system(ffmpeg_call)

if __name__ == '__main__':
    # parse command line arguments
    if len(sys.argv) == 1:
        print('Invalid arguments given! Missing simulation directory path!')
        sys.exit(1)
    casedir = sys.argv[1]
    try:
        frame_rate = sys.argv[2]
    except IndexError:
        frame_rate = 60

    # generate pngs
    make_pngs(casedir)

    # make video using ffmpeg
    pngs2mp4(casedir, 'wearenumberone.mp4', frame_rate=frame_rate)
