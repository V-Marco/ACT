import sys
sys.path.append("../")
sys.path.append("../Modules/")

import analysis
from logger import Logger
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import traceback

# TODO: 
# (1) control for bursts

# https://github.com/dbheadley/InhibOnDendComp/blob/master/src/mean_dendevt.py
def _plot_sta(
          sta, 
          quantiles, 
          title,
          xlabel_spike_type,
          ylabel_ed_from) -> plt.figure:
    
    x_ticks = np.arange(0, 50, 5)
    x_tick_labels = ['{}'.format(i) for i in np.arange(-50, 50, 10)]
     
    fig = plt.figure(figsize = (10, 5))
    plt.imshow(sta, cmap = sns.color_palette("coolwarm", as_cmap = True))
    plt.title(title)
    plt.xticks(ticks = x_ticks - 0.5, labels = x_tick_labels)
    plt.xlabel(f'Time w.r.t. {xlabel_spike_type} spikes (ms)')
    plt.yticks(ticks = np.arange(11) - 0.5, labels = np.round(quantiles, 3))
    plt.ylabel(f"Elec. dist. quantile (from {ylabel_ed_from})")
    plt.colorbar(label = 'Percent change from mean')
    return fig

def _compute_sta_for_each_train_in_a_list(list_of_trains, spikes) -> np.ndarray:

    parameters = analysis.DataReader.load_parameters(sim_directory)

    stas = []
    for train in list_of_trains:
        if len(train) == 0: 
            stas.append(np.zeros((1, 50)))
            continue
        cont_train = np.zeros(parameters.h_tstop)
        cont_train[train] = 1

        # Skip spikes that are in the beginning of the trace
        cont_train[:parameters.skip] = 0
        sta = analysis.SummaryStatistics.spike_triggered_average(cont_train.reshape((1, -1)), spikes, 50)

        # Normalize by average
        sta = (sta - np.mean(cont_train)) / (np.mean(cont_train) + 1e-15)
        stas.append(sta)

    stas = np.concatenate(stas)
    return stas

def _map_stas_to_quantiles_and_plot(
        sta, 
        spikes, 
        section,
        elec_dist_from,
        title,
        xlabel_spike_type, 
        indexes = None) -> None:
    
    if section not in ["apic", "dend"]: raise ValueError
    if elec_dist_from not in ["soma", "nexus"]: raise ValueError
    
    elec_dist = pd.read_csv(os.path.join(sim_directory, f"elec_distance_{elec_dist_from}.csv"))
    morph = pd.read_csv(os.path.join(sim_directory, "segment_data.csv"))

    if indexes is not None:
        elec_dist = elec_dist.iloc[indexes, :]
        morph = morph.iloc[indexes, :]

    quantiles = analysis.SummaryStatistics.get_quantiles_based_on_elec_dist(
        morph = morph,
        elec_dist = elec_dist,
        spikes = spikes,
        section = section
    )

    sta_binned = analysis.SummaryStatistics.bin_matrix_to_quantiles(
        matrix = sta,
        quantiles = quantiles, 
        var_to_bin = elec_dist
    )

    fig = _plot_sta(sta_binned, quantiles, title, xlabel_spike_type, elec_dist_from)
    plt.show()
    if save:
        fig.savefig(os.path.join(sim_directory, f"{title}_{xlabel_spike_type}_spikes_{elec_dist_from}_ed.png"), dpi = fig.dpi)

def _analyze_Na():

    gnaTa = analysis.DataReader.read_data(sim_directory, "gNaTa_t_NaTa_t")
    soma_spikes = analysis.DataReader.read_data(sim_directory, "soma_spikes")
    v = analysis.DataReader.read_data(sim_directory, "v")
    
    Na_spikes = []
    for i in range(len(gnaTa)):
        spikes, _ = analysis.VoltageTrace.get_Na_spikes(gnaTa[i], 0.001 / 1000, soma_spikes, 2, v[i], v[0])
        Na_spikes.append(spikes)

    sta = _compute_sta_for_each_train_in_a_list(Na_spikes, soma_spikes)
    for section in ["apic", "dend"]:
        for elec_dist in ["soma", "nexus"]:
            try:
                _map_stas_to_quantiles_and_plot(
                    sta = sta, 
                    spikes = Na_spikes, 
                    section = section,
                    elec_dist_from = elec_dist,
                    title = f"Na-{section}",
                    xlabel_spike_type = "soma")
            except:
                print(section, elec_dist)
                print(traceback.format_exc())
                continue

def _analyze_Ca():

    lowery = 500
    uppery = 1500

    v = analysis.DataReader.read_data(sim_directory, "v")
    ica = analysis.DataReader.read_data(sim_directory, "ica")
    soma_spikes = analysis.DataReader.read_data(sim_directory, "soma_spikes")

    for section in ["apic", "dend"]:
        try:
            seg_data = pd.read_csv(os.path.join(sim_directory, "segment_data.csv"))
            indexes = seg_data[(seg_data["section"] == section) & (seg_data["pc_1"] > lowery) & (seg_data["pc_1"] < uppery)].index

            Ca_spikes = []
            for i in indexes:
                left_bounds, _, _ = analysis.VoltageTrace.get_Ca_spikes(v[i], -40, ica[i])
                Ca_spikes.append(left_bounds)
            
            sta = _compute_sta_for_each_train_in_a_list(Ca_spikes, soma_spikes)
            for elec_dist in ["soma", "nexus"]:
                try:
                    _map_stas_to_quantiles_and_plot(
                        sta = sta,
                        spikes = Ca_spikes, 
                        section = section,
                        elec_dist_from = elec_dist,
                        title = f"Ca-{section}",
                        xlabel_spike_type = "soma",
                        indexes = indexes)
                except:
                    print(section, elec_dist)
                    print(traceback.format_exc())
                    continue
        except:
            print(section)
            print(traceback.format_exc()) 
            continue

def _analyze_NMDA():

    v = analysis.DataReader.read_data(sim_directory, "v")
    inmda = analysis.DataReader.read_data(sim_directory, "i_NMDA")
    ica = analysis.DataReader.read_data(sim_directory, "ica")
    soma_spikes = analysis.DataReader.read_data(sim_directory, "soma_spikes")

    for section in ["apic", "dend"]:

        seg_data = pd.read_csv(os.path.join(sim_directory, "segment_data.csv"))
        indexes = seg_data[seg_data["section"] == section].index

        try:
            NMDA_spikes = []
            for i in indexes:
                left_bounds, _, _ = analysis.VoltageTrace.get_NMDA_spikes(v[i], -40, inmda[i])
                NMDA_spikes.append(left_bounds)
        except:
            print(section)
            print(traceback.format_exc()) 
            continue # There are no NMDA spikes

        try:
            # Soma
            sta = _compute_sta_for_each_train_in_a_list(NMDA_spikes, soma_spikes)
            for elec_dist in ["soma", "nexus"]:
                try:
                    _map_stas_to_quantiles_and_plot(
                        sta = sta,
                        spikes = NMDA_spikes, 
                        section = section,
                        elec_dist_from = elec_dist,
                        title = f"NMDA-{section}",
                        xlabel_spike_type = "soma",
                        indexes = indexes)
                except:
                    print(section, elec_dist)
                    print(traceback.format_exc()) 
                    continue
        except:
            print(section, elec_dist)
            print(traceback.format_exc()) 
            pass
        
        try:
            # Ca
            Ca_spikes = []
            for i in indexes:
                left_bounds, _, _ = analysis.VoltageTrace.get_Ca_spikes(v[i], -40, ica[i])
                Ca_spikes.extend(left_bounds)
            Ca_spikes = np.sort(np.unique(Ca_spikes))

            sta = _compute_sta_for_each_train_in_a_list(NMDA_spikes, Ca_spikes)
            for elec_dist in ["soma", "nexus"]:
                try:
                    _map_stas_to_quantiles_and_plot(
                        sta = sta,
                        spikes = NMDA_spikes, 
                        section = section,
                        elec_dist_from = elec_dist,
                        title = f"NMDA-{section}",
                        xlabel_spike_type = "Ca",
                        indexes = indexes)
                except:
                    print(section, elec_dist)
                    print(traceback.format_exc()) 
                    continue
        except:
            print(section)
            print(traceback.format_exc()) 
            pass
        

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

