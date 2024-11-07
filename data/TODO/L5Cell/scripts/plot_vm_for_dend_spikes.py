import sys
sys.path.append("../")
sys.path.append("../Modules/")

import analysis
from logger import Logger
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import pandas as pd
import os
import traceback
import random


# ADD electrotonic distance from soma to legend.
# UPDATE generate one figure per somatic spike and accompanying dend spikes
# add what stars mean.
def plot_voltage_for_segments_where_spikes_occur(
    seg_data,
    v,
    soma_spikes,
    dend_spike_start_times,
    title,
    section,
    min_time_index=0,
    max_time_index=2000,
    time_indices_before_soma_spike=50,
    time_indices_after_soma_spike=50,
    max_soma_spikes_plotted=1,
    max_segments_with_dend_spikes_per_soma_spike=100,
    save=True,
    sim_directory='',
    include_legend=True,
    use_varied_colors=True):
    """
    Plots voltage for segments where spikes occur.

    Parameters:
    ... [other parameters] ...
    include_legend : bool, optional
        Include a legend in the plot to identify segment indices.
    use_varied_colors : bool, optional
        Use varied colors for dendritic voltages.
    """

    if title == 'Ca':
        indexes = seg_data[(seg_data["section"] == "apic")].index
    else:
        indexes = seg_data[(seg_data["section"] == section)].index
        
    print(seg_data.keys())
    print(f"{title+section} indexes: {indexes}")

    fig, ax = plt.figure(figsize=(15, 10)), plt.gca()

    # Generate a list of distinct colors if varied colors are to be used
    if use_varied_colors:
        palette_name = 'tab20'
        colors = list(plt.get_cmap(palette_name).colors)  # Convert to list before shuffling
        random.shuffle(colors)
    # Randomly select soma spikes, respecting the maximum limit
    random_soma_spikes = random.sample(list(soma_spikes[0][1:-1]), min(len(soma_spikes[0][1:-1]), max_soma_spikes_plotted))

    for soma_spike in random_soma_spikes:
        min_index = int(max(soma_spike - time_indices_before_soma_spike, 0))
        max_index = int(min(soma_spike + time_indices_after_soma_spike, len(v[0])))

        # Check if the range around the soma spike is sufficient
        if (max_index - min_index) != (time_indices_before_soma_spike + time_indices_after_soma_spike):
            continue

        eligible_segments = []
        for seg_index in indexes:
            seg_dend_spikes = dend_spike_start_times[seg_index]
            if any(min_index < spike_time < max_index for spike_time in seg_dend_spikes):
                eligible_segments.append(seg_index)

        # Randomly select segments to plot, respecting the maximum limit
        segments_to_plot = random.sample(eligible_segments, min(len(eligible_segments), max_segments_with_dend_spikes_per_soma_spike))

    for i, seg_index in enumerate(segments_to_plot):
        seg_dend_spikes = dend_spike_start_times[seg_index]
        color = colors[i % len(colors)] if use_varied_colors else 'blue'
        ax.plot(v[seg_index][min_index:max_index], color=color, linewidth=0.25, label=f'Segment {seg_index}' if use_varied_colors else None)
        for spike_time in seg_dend_spikes:
            if min_index < spike_time < max_index:
                spike_label = 'Dendritic Spike' if f'Dendritic Spike' not in ax.get_legend_handles_labels()[1] else None
                ax.plot(spike_time - min_index, v[seg_index][spike_time], '*', markersize=10, color=color, label=spike_label)
    
    if len(segments_to_plot) > 0:
        ax.plot(v[0][min_index:max_index], color='black', linewidth=1.0, label='Soma')
        soma_spike_label = 'Somatic Spike' if 'Somatic Spike' not in ax.get_legend_handles_labels()[1] else None
        ax.plot(soma_spike - min_index, max(v[0][min_index:max_index]), '*', markersize=10, color='black', label=soma_spike_label)
    
    ax.set_title('Vm ' + title + section)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel("Voltage")
    
    if include_legend:
        ax.legend()

    plt.show()

    if save:
        fig.savefig(os.path.join(sim_directory, f"Vm_{title+'_'+section}.png"), dpi=fig.dpi)
    

# old code for reference. Can update current code to include stop_times
#def plot_voltage_for_segments_where_spikes_occur(v, segment_indices, spike_start_times, spike_stop_times, spike_type, output_folder, color, max_indices=2000):
#    plt.figure(figsize=(15, 10))
#
#    plt.plot(segment_manager.segments[0].v[:max_indices], label=f'{segment_manager.segments[0].seg}', color='black', alpha=0.6)
#    soma_spike_times = [time for time in segment_manager.soma_spiketimes if time < max_indices]
#            # Mark each spike start time with a green star
#    for soma_spike_time in soma_spike_times:
#        plt.plot(soma_spike_time, segment_manager.segments[0].v[soma_spike_time], '*', markersize=10, color='Black')
#
#
#    for i in np.unique(segment_indices):
#        seg = segment_manager.segments[seg_index]
#
#        # Plot the entire voltage trace up to max_indices for each segment
#        plt.plot(seg.v[:max_indices], label=f'{seg.seg}', color=color, alpha=0.6)
#
#        # Select spike start and stop times for the current segment, ensuring they are within max_indices
#        seg_spike_start_times = [time for time in spike_start_times[seg_index] if time < max_indices]
#        seg_spike_stop_times = [time for time in spike_stop_times[seg_index] if time < max_indices]
#
#        # Mark each spike start time with a green star
#        for spike_time in seg_spike_start_times:
#            plt.plot(spike_time, seg.v[spike_time], '*', markersize=10, color='green')
#
#        # Mark each spike stop time with a red rectangle
#        for spike_time in seg_spike_stop_times:
#            plt.plot(spike_time, seg.v[spike_time], 's', markersize=10, color='red')
#
#    plt.xlabel('Index')
#    plt.ylabel('Voltage')
#    plt.title(f'{spike_type} Spikes')
#    plt.legend()
#    plt.savefig(os.path.join(output_folder, f'{spike_type}_spikes.png'))
#    plt.close()
    
def _analyze_Na():

    gnaTa = analysis.DataReader.read_data(sim_directory, "gNaTa_t_NaTa_t")
    v = analysis.DataReader.read_data(sim_directory, "v")
    soma_spikes = analysis.DataReader.read_data(sim_directory, "soma_spikes")
    v = analysis.DataReader.read_data(sim_directory, "v")
    
    Na_spikes = []
    for i in range(len(gnaTa)):
        spike_start_times, _ = analysis.VoltageTrace.get_Na_spikes(gnaTa[i], 0.001 / 1000, soma_spikes, 2, v[i], v[0])
        Na_spikes.append(spike_start_times)
    seg_data = pd.read_csv(os.path.join(sim_directory, "segment_data.csv"))

    # compute and plot
    plot_voltage_for_segments_where_spikes_occur(seg_data, v, soma_spikes, Na_spikes, 'Na',section='dend')
    plot_voltage_for_segments_where_spikes_occur(seg_data, v, soma_spikes, Na_spikes, 'Na',section='apic')
    #sta = _compute_sta_for_each_train_in_a_list(Na_spikes)
    #_map_stas_to_quantiles_and_plot(sta, Na_spikes, "dend", "Na-dend")
    #_map_stas_to_quantiles_and_plot(sta, Na_spikes, "apic", "Na-apic")

def _analyze_Ca():

    lowery = 500
    uppery = 1500

    seg_data = pd.read_csv(os.path.join(sim_directory, "segment_data.csv"))
    #indexes = seg_data[(seg_data["section"] == "apic") & (seg_data["pc_1"] > lowery) & (seg_data["pc_1"] < uppery)].index

    v = analysis.DataReader.read_data(sim_directory, "v")
    ica = analysis.DataReader.read_data(sim_directory, "ica")

    Ca_spikes = []
    for i in range(len(v)):#for i in indexes:
        left_bounds, _, _ = analysis.VoltageTrace.get_Ca_spikes(v[i], -40, ica[i])
        Ca_spikes.append(left_bounds)
        
    #compute and plot
    #plot_voltage_for_segments_where_spikes_occur(v, soma_spikes, Ca_spikes, 'Ca',section='dend')
    plot_voltage_for_segments_where_spikes_occur(seg_data, v, soma_spikes, Ca_spikes, 'Ca',section='apic')


def _analyze_NMDA():

    seg_data = pd.read_csv(os.path.join(sim_directory, "segment_data.csv"))
    #indexes = seg_data[(seg_data["section"] == "apic") | (seg_data["section"] == "dend")].index

    v = analysis.DataReader.read_data(sim_directory, "v")
    inmda = analysis.DataReader.read_data(sim_directory, "i_NMDA")

    NMDA_spikes = []
    for i in range(len(v)):#for i in indexes:
        left_bounds, _, _ = analysis.VoltageTrace.get_NMDA_spikes(v[i], -40, inmda[i])
        NMDA_spikes.append(left_bounds)

    #compute and plot
    plot_voltage_for_segments_where_spikes_occur(seg_data, v, soma_spikes, NMDA_spikes, 'NMDA',section='dend')
    plot_voltage_for_segments_where_spikes_occur(seg_data, v, soma_spikes, NMDA_spikes, 'NMDA',section='apic')

    
if __name__ == "__main__":

    if "-d" in sys.argv:
        sim_directory = sys.argv[sys.argv.index("-d") + 1] # (global)
    else:
        raise RuntimeError

    # Save figures or just show them
    save = "-s" in sys.argv # (global)

    logger = Logger()

    soma_spikes = analysis.DataReader.read_data(sim_directory, "soma_spikes")
    parameters = analysis.DataReader.load_parameters(sim_directory)
    logger.log(f"Soma firing rate: {round(soma_spikes.shape[1] * 1000 / parameters.h_tstop, 2)} Hz")

    try:
        logger.log("Analyzing Na.")
        _analyze_Na()
    except Exception:
        print(traceback.format_exc())
    
    try:
        logger.log("Analyzing Ca.")
        _analyze_Ca()
    except Exception:
        print(traceback.format_exc())

    try:
        logger.log("Analyzing NMDA.")
        _analyze_NMDA()
    except Exception:
        print(traceback.format_exc())