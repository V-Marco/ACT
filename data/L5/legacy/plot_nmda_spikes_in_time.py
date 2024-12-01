import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
import os

from Modules.segment import SegmentManager
import constants

from matplotlib.widgets import Slider

output_folder = "output/2023-08-30_14-50-37_seeds_130_90L5PCtemplate[0]_196nseg_108nbranch_29543NCs_29543nsyn"


def main():

    random_state = np.random.RandomState(123)

    step_size = int(constants.save_every_ms / constants.h_dt) # Timestamps
    steps = range(step_size, int(constants.h_tstop / constants.h_dt) + 1, step_size) # Timestamps

    sm = SegmentManager(output_folder, steps = steps, dt = constants.h_dt)

    nmda_lower_bounds, upper_bounds, _, _, _, _ = sm.get_ca_nmda_lower_bounds_durations_and_peaks(lowery = None, uppery = None, random_state = random_state)
    
    one_branch_seg_inds = [152, 146, 145, 135, 121] # [121, 122, 130, 134]
    one_branch_segs = []
    one_brain_lb = []
    one_brain_ub = []

    spikes = np.zeros((len(one_branch_seg_inds), int(constants.h_tstop)))

    for ind in one_branch_seg_inds:
        one_branch_segs.append(sm.segments[ind])
        one_brain_lb.append(nmda_lower_bounds[ind])
        one_brain_ub.append(upper_bounds[ind])

    for seg_idx in range(len(one_brain_lb)):
        for bound_indx in range(len(one_brain_lb[seg_idx])):
            spikes[seg_idx, int(one_brain_lb[seg_idx][bound_indx] * constants.h_dt) : int(one_brain_ub[seg_idx][bound_indx] * constants.h_dt)] = 1

    fig, ax = plt.subplots(2, 1, figsize = (20, 5))
    for seg in one_branch_segs:
        ax[0].scatter(seg.p0_5_x3d, seg.p0_5_y3d, c = "tab:blue")
    ax[0].set_xlabel("p0_5_x3d")
    ax[0].set_ylabel("p0_5_y3d")

    N = 100
    def mat(pos):
        pos = int(pos)
        ax[1].clear()
        if pos + N > spikes.shape[1]: 
            n = spikes.shape[1] - pos
        else:
            n = N
        ax[1].matshow(spikes[:, pos : pos + n])
        ax[1].set_xticks(np.arange(N), labels = np.arange(pos + 1, pos + n + 1), rotation = 90, fontsize = 8)
        ax[1].set_yticks(np.arange(len(one_branch_seg_inds)), labels = one_branch_seg_inds, fontsize = 8)

        ax[1].set_xlabel("ms")
        ax[1].set_ylabel("seg_index")
    
    barpos = plt.axes([0.18, 0.05, 0.55, 0.03], facecolor = "skyblue")
    slider = Slider(barpos, 'ms', 0, spikes.shape[1] - N, valinit = 0)
    slider.on_changed(mat)
    mat(0)
    plt.show()

if __name__ == "__main__":
    main()