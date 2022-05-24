# generate a video file from given data.

import os
import sys
import argparse

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from rpcf.simulation import SimulationResults

def make_pngs(casedir, vid_range=None):
    # make sure directories exist
    if not os.path.isdir(os.path.join(os.getcwd(), 'videos/', os.path.basename(casedir[:-1]))):
        os.mkdir(os.path.join(os.getcwd(), 'videos/', os.path.basename(casedir[:-1])))

    # unpack simulation data
    sim = SimulationResults(casedir)

    # set time range of video
    if vid_range is None:
        vid_range = range(len(sim.t))
    else:
        vid_range = range(np.where(sim.t == vid_range[0])[0][0], np.where(sim.t == vid_range[1])[0][0])
    
    # define contour levels
    maxval, minval = 0, 0
    for i in vid_range:
        snap = sim[sim.t[i]]
        max_at_t = np.amax(snap.omega.data)
        min_at_t = np.amin(snap.omega.data)
        if max_at_t > maxval:
            maxval = max_at_t
        if min_at_t < minval:
            minval = min_at_t
    levels = np.linspace(minval, maxval, 100)

    mpl.rcParams['image.cmap'] = 'seismic'
    for i in vid_range:
        print(f'Plotting: {i}, {sim.t[i]}')
        snap = sim[sim.t[i]]
        plt.clf()
        plt.contourf(snap.omega.z, snap.omega.y, snap.omega.data, levels, extend="both", vmin=minval, vmax=maxval)
        plt.gca().set_aspect(1)
        plt.xticks([0, 2, 4, 6, 8])
        plt.yticks([-1, 0, 1])
        plt.xlabel('$z$')
        plt.ylabel('$y$')
        plt.savefig(os.path.join(os.getcwd(), 'videos/', os.path.basename(casedir[:-1]), 'frame%06d.png' % i), dpi=1000, bbox_inches='tight')

def pngs2mp4(casedir, video_name, frame_rate):
    video_dir = os.path.join(os.getcwd(), 'videos/', os.path.basename(casedir[:-1]))
    ffmpeg_call = f"ffmpeg -r {frame_rate} -pattern_type glob -i '{video_dir}/frame*.png' -vcodec libx264 {video_dir}/{video_name}"
    os.system(ffmpeg_call)

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('casedir', type=str)
    parser.add_argument('--frame_rate', '-r', type=int)
    parser.add_argument('--vid_range', '-v', type=str)
    parser.add_argument('--vid_name', '-n', type=str)
    parser.add_argument('--no_vid', action='store_false')
    args = parser.parse_args()
    if args.frame_rate is None:
        args.frame_rate = 60
    if args.vid_range is not None:
        args.vid_range = list(map(float, args.vid_range.split('-')))
    if args.vid_name is None:
        args.vid_name = 'wearenumberone.mp4'

    # generate pngs
    make_pngs(args.casedir, vid_range=args.vid_range)

    # make video using ffmpeg
    if args.no_vid:
        pngs2mp4(args.casedir, args.vid_name, frame_rate=args.frame_rate)
