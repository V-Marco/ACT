# Set the path
import sys
sys.path.append("../../")

from act.cell_model import ACTCellModel
from act.simulator import ACTSimulator
from act.types import SimulationParameters, ConstantCurrentInjection, FilterParameters, ConductanceOptions, OptimizationParameters
import act.data_processing as dp
from act.module import ACTModule

import numpy as np
import matplotlib.pyplot as plt

from act.passive import ACTPassiveModule
from act.segregation import ACTSegregator

from sklearn.metrics import mean_absolute_error
from act.metrics import pp_error

target_cell = ACTCellModel(
    path_to_hoc_file="/home/ubuntu/ACT/data/LAA/orig/target_template.hoc",
    path_to_mod_files="/home/ubuntu/ACT/data/LAA/orig/modfiles",
    cell_name="Cell_A",
    passive=[],
    active_channels=["gbar_nap",
                     "gmbar_im", 
                     "gbar_na3",
                     "gkdrbar_kdr", 
                     "gcabar_cadyn", 
                     "gsAHPbar_sAHP", 
                     "gkabar_kap",
                     "ghdbar_hd",
                     "glbar_leak"]
)

target_g = np.array([0.0003, 0.002, 0.03, 0.003, 6e-5, 0.009, 0.000843, 2.3e-05, 3.5e-5])

# Set simulations
simulator = ACTSimulator(output_folder_name = "output")

sim_params = SimulationParameters(
    sim_name = "target",
    sim_idx = 0,
    h_v_init=-70,
    h_celsius = 6.3,
    h_dt = 0.1,
    h_tstop = 1000,
    CI = [ConstantCurrentInjection(amp = -0.2, dur = 700, delay = 100)])

simulator.submit_job(target_cell, sim_params)
simulator.run_jobs(1)

passive_trace = np.load("output/target/out_0.npy")[:, 0]
plt.plot(passive_trace[::10])
plt.savefig("output/Passive_LAA.png")

target_gpp = ACTPassiveModule.compute_gpp(passive_trace, 0.1, 100, 700, -0.2)
target_gpp

# Set simulations
simulator = ACTSimulator(output_folder_name = "output")

for sim_idx, amp_value in enumerate([0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.1, 0.3, 0.5, 0.7, 1.0, 3.0, 5.0, 7.0, 9.0]):
    sim_params = SimulationParameters(
        sim_name = "target",
        sim_idx = sim_idx,
        h_v_init=-70,
        h_celsius = 6.3,
        h_dt = 0.1,
        h_tstop = 1000,
        CI = [ConstantCurrentInjection(amp = amp_value, dur = 700, delay = 100)])

    simulator.submit_job(target_cell, sim_params)

simulator.run_jobs(3)

# Combine simulated traces into one dataset for convenience
dp.combine_data("output/target")

# Plot the traces and the FI curve
simulated_data = np.load("output/target/combined_out.npy") # 3 x 10000 x 4; (n_sim x time x [V, I, g, lto_hto])

fig, ax = plt.subplots(5, 3, figsize = (10, 12))
ax = ax.flatten()

for axid, amp in enumerate([0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.1, 0.3, 0.5, 0.7, 1.0, 3.0, 5.0, 7.0, 9.0]):
    ax[axid].plot(simulated_data[axid, ::10, 0])
    ax[axid].set_xlabel("Time (ms)")
    ax[axid].set_title(f"CI = {amp} nA")

ax[0].set_ylabel("Voltage (mV)")

plt.tight_layout()
plt.show()
plt.savefig("output/V_Traces.png")

simulated_data = np.load("output/target/combined_out.npy")

f = []
for trace_id in range(len(simulated_data)):
    f.append(len(dp.find_events(simulated_data[trace_id, ::10, 0].flatten())))

plt.plot([0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.1, 0.3, 0.5, 0.7, 1.0, 3.0, 5.0, 7.0, 9.0], f)
plt.xlabel("CI (nA)")
plt.ylabel("Frequency (Hz)")
plt.title("FI Curve")

random_state = np.random.RandomState(123)
target_values = np.array([2.5, 0.0003, 0.002, 0.03, 0.003, 6e-5, 0.009, 0.000843, 2.3e-05, 3.5e-5, -70]) 

#Cm, gbar_nap, gmbar_im, gbar_na3, gkdrbar_kdr, gcabar_cadyn, gsAHPbar_sAHP, gkabar_kap, ghdbar_hd, glbar_leak, el_leak

target_values = target_values + random_state.normal(0, np.abs(target_values * 0.1))
target_values

# Define the train cell
train_cell = ACTCellModel(
    path_to_hoc_file="/home/ubuntu/ACT/data/LAA/orig/template.hoc",
    path_to_mod_files="/home/ubuntu/ACT/data/LAA/orig/modfiles",
    cell_name="Cell_A",
    passive=[],
    active_channels=["gbar_nap",
                     "gmbar_im", 
                     "gbar_na3",
                     "gkdrbar_kdr", 
                     "gcabar_cadyn", 
                     "gsAHPbar_sAHP", 
                     "gkabar_kap",
                     "ghdbar_hd",
                     "glbar_leak"]
)

# Set simulations
simulator = ACTSimulator(output_folder_name = "output")

sim_params = SimulationParameters(
    sim_name = "orig",
    sim_idx = 0,
    h_v_init=-70,
    h_celsius = 6.3,
    h_dt = 0.1,
    h_tstop = 1000,
    CI = [ConstantCurrentInjection(amp = -0.2, dur = 700, delay = 100)])

simulator.submit_job(train_cell, sim_params)
simulator.run_jobs(1)

passive_trace = np.load("output/orig/out_0.npy")[:, 0]
ACTPassiveModule.compute_gpp(passive_trace, 0.1, 100, 700, -0.2)

# Set simulations
simulator = ACTSimulator(output_folder_name = "output")

for sim_idx, amp_value in enumerate([0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.1, 0.3, 0.5, 0.7, 1.0, 3.0, 5.0, 7.0, 9.0]):
    sim_params = SimulationParameters(
        sim_name = "orig",
        sim_idx = sim_idx,
        h_v_init=-70,
        h_celsius = 6.3,
        h_dt = 0.1,
        h_tstop = 1000,
        CI = [ConstantCurrentInjection(amp = amp_value, dur = 700, delay = 100)])

    simulator.submit_job(train_cell, sim_params)

simulator.run_jobs(3)

# Combine simulated traces into one dataset for convenience
dp.combine_data("output/orig")

simulated_data = np.load("output/orig/combined_out.npy")

f = []
for trace_id in range(len(simulated_data)):
    f.append(len(dp.find_events(simulated_data[trace_id, ::10, 0].flatten())))

plt.plot([0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.1, 0.3, 0.5, 0.7, 1.0, 3.0, 5.0, 7.0, 9.0], f)
plt.xlabel("CI (nA)")
plt.ylabel("Frequency (Hz)")
plt.title("FI Curve")
plt.savefig("output/FI_Before_Tuning.png")

# Parameter ranges as if provided by the user
gbar_nap = 3.299204e-4
gmbar_im = 2.056596e-3 
gbar_na3 =2.548112e-2
gkdrbar_kdr = 2.82642e-3 
gcabar_cadyn =6.99086e-5 
gsAHPbar_sAHP = 6.81599e-3 
gkabar_kap = 8.068427e-4 
ghdbar_hd = 2.591165e-5 
glbar_leak = 3.196641e-5 

random_state = np.random.RandomState(123)

gbar_nap_range = (gbar_nap - random_state.uniform(0, gbar_nap / 2), gbar_nap + random_state.uniform(0, gbar_nap / 2))
gmbar_im_range = (gmbar_im - random_state.uniform(0, gmbar_im / 2), gmbar_im + random_state.uniform(0, gmbar_im / 2))
gbar_na3_range = (gbar_na3 - random_state.uniform(0, gbar_na3 / 2), gbar_na3 + random_state.uniform(0, gbar_na3 / 2))
gkdrbar_kdr_range = (gkdrbar_kdr - random_state.uniform(0, gkdrbar_kdr / 2), gkdrbar_kdr + random_state.uniform(0, gkdrbar_kdr / 2))
gcabar_cadyn_range = (gcabar_cadyn - random_state.uniform(0, gcabar_cadyn / 2), gcabar_cadyn + random_state.uniform(0, gcabar_cadyn / 2))
gsAHPbar_sAHP_range = (gsAHPbar_sAHP - random_state.uniform(0, gsAHPbar_sAHP / 2), gsAHPbar_sAHP + random_state.uniform(0, gsAHPbar_sAHP / 2))
gbar_kap_range = (gkabar_kap - random_state.uniform(0, gkabar_kap / 2), gkabar_kap + random_state.uniform(0, gkabar_kap / 2))
ghdbar_hd_range = (ghdbar_hd - random_state.uniform(0, ghdbar_hd / 2), ghdbar_hd + random_state.uniform(0, ghdbar_hd / 2))
glbar_leak_range = (glbar_leak - random_state.uniform(0, glbar_leak / 2), glbar_leak + random_state.uniform(0, glbar_leak / 2))


print(f"Nap: {gbar_nap_range}")
print(f"Nap: {gmbar_im_range}")
print(f"Na3: {gbar_na3_range}")
print(f"Kdr: {gkdrbar_kdr_range}")
print(f"Cadyn: {gcabar_cadyn_range}")
print(f"sAHP: {gsAHPbar_sAHP_range}")
print(f"Kap: {gbar_kap_range}")
print(f"Hd: {ghdbar_hd_range}")
print(f"Leak: {glbar_leak_range}")

# Possibly adjsut
train_cell = ACTCellModel(
    path_to_hoc_file="/home/ubuntu/ACT/data/LAA/orig/template.hoc",
    path_to_mod_files="/home/ubuntu/ACT/data/LAA/orig/modfiles",
    cell_name="Cell_A",
    passive=[],
    active_channels=["gbar_nap",
                     "gmbar_im", 
                     "gbar_na3",
                     "gkdrbar_kdr", 
                     "gcabar_cadyn", 
                     "gsAHPbar_sAHP", 
                     "gkabar_kap",
                     "ghdbar_hd",
                     "glbar_leak"]
)

from multiprocessing import Pool, cpu_count
sim_params = SimulationParameters(
        sim_name = "cell",
        sim_idx = 0,
        h_v_init=-70,
        h_celsius = 6.3,
        h_dt = 0.1,
        h_tstop = 1000)

optim_params = OptimizationParameters(
    conductance_options = [
        ConductanceOptions(variable_name = "gbar_nap", low = gbar_nap_range[0], high = gbar_nap_range[1], n_slices = 3),
        ConductanceOptions(variable_name = "gmbar_im", low = gmbar_im_range[0], high = gmbar_im_range[1], n_slices = 3),
        ConductanceOptions(variable_name = "gbar_na3", low = gbar_na3_range[0], high = gbar_na3_range[1], n_slices = 3),
        ConductanceOptions(variable_name = "gkdrbar_kdr", low = gkdrbar_kdr_range[0], high = gkdrbar_kdr_range[1], n_slices = 3),
        ConductanceOptions(variable_name = "gcabar_cadyn", low = gcabar_cadyn_range[0], high = gcabar_cadyn_range[1], n_slices = 3),  
        ConductanceOptions(variable_name = "gsAHPbar_sAHP", low = gsAHPbar_sAHP_range[0], high = gsAHPbar_sAHP_range[1], n_slices = 3),
        ConductanceOptions(variable_name = "gkabar_kap", low = gbar_kap_range[0], high = gbar_kap_range[1], n_slices = 3),
        ConductanceOptions(variable_name = "ghdbar_hd", low = ghdbar_hd_range[0], high = ghdbar_hd_range[1], n_slices = 3),
        ConductanceOptions(variable_name = "glbar_leak", low = glbar_leak_range[0], high = glbar_leak_range[1], n_slices = 3)
    ],
    CI_options = [
        ConstantCurrentInjection(amp = 0.0, dur = 700, delay = 100),
        ConstantCurrentInjection(amp = 0.02, dur = 700, delay = 100),
        ConstantCurrentInjection(amp = 0.03, dur = 700, delay = 100),
        ConstantCurrentInjection(amp = 0.04, dur = 700, delay = 100),
        ConstantCurrentInjection(amp = 0.05, dur = 700, delay = 100),
        ConstantCurrentInjection(amp = 0.06, dur = 700, delay = 100),
        ConstantCurrentInjection(amp = 0.1, dur = 700, delay = 100),
        ConstantCurrentInjection(amp = 0.3, dur = 700, delay = 100),
        ConstantCurrentInjection(amp = 0.5, dur = 700, delay = 100),
        ConstantCurrentInjection(amp = 0.7, dur = 700, delay = 100),
        ConstantCurrentInjection(amp = 1.0, dur = 700, delay = 100),
        ConstantCurrentInjection(amp = 3.0, dur = 700, delay = 100),
        ConstantCurrentInjection(amp = 5.0, dur = 700, delay = 100),
        ConstantCurrentInjection(amp = 7.0, dur = 700, delay = 100),
        ConstantCurrentInjection(amp = 9.0, dur = 700, delay = 100)
    ],
    filter_parameters = FilterParameters(
        saturation_threshold = -55,
        window_of_inspection = (100, 700)
    ),
    n_cpus=30
)

m = ACTModule(
    name = "orig",
    cell = train_cell,
    simulation_parameters = sim_params,
    optimization_parameters = optim_params,
    target_file = "output/target/combined_out.npy"
)

m.run()

orig_g = np.array(list(m.cell.prediction.values()))

# Test g error
mean_absolute_error(target_g, orig_g)

train_cell = ACTCellModel(
    path_to_hoc_file="/home/ubuntu/ACT/data/LAA/orig/template.hoc",
    path_to_mod_files="/home/ubuntu/ACT/data/LAA/orig/modfiles",
    cell_name="Cell_A",
    passive=[],
    active_channels=["gbar_nap",
                     "gmbar_im", 
                     "gbar_na3",
                     "gkdrbar_kdr", 
                     "gcabar_cadyn", 
                     "gsAHPbar_sAHP", 
                     "gkabar_kap",
                     "ghdbar_hd",
                     "glbar_leak"]
)

# Set simulations
simulator = ACTSimulator(output_folder_name = "output")

sim_params = SimulationParameters(
    sim_name = "laa_orig_after",
    sim_idx = 0,
    h_v_init=-70,
    h_celsius = 6.3,
    h_dt = 0.1,
    h_tstop = 1000,
    CI = [ConstantCurrentInjection(amp = -0.2, dur = 700, delay = 100)])

train_cell.set_g_bar(["gbar_nap",
                     "gmbar_im", 
                     "gbar_na3",
                     "gkdrbar_kdr", 
                     "gcabar_cadyn", 
                     "gsAHPbar_sAHP", 
                     "gkabar_kap",
                     "ghdbar_hd",
                     "glbar_leak"], orig_g)

simulator.submit_job(train_cell, sim_params)
simulator.run_jobs(1)

passive_trace = np.load("output/laa_orig_after/out_0.npy")[:, 0]
ACTPassiveModule.compute_gpp(passive_trace, 0.1, 100, 700, -0.2)

orig_gpp = ACTPassiveModule.compute_gpp(passive_trace, 0.1, 100, 700, -0.2)
orig_gpp

pp_error(target_gpp, orig_gpp)

train_cell = ACTCellModel(
    path_to_hoc_file="/home/ubuntu/ACT/data/LAA/orig/template.hoc",
    path_to_mod_files="/home/ubuntu/ACT/data/LAA/orig/modfiles",
    cell_name="Cell_A",
    passive=[],
    active_channels=["gbar_nap",
                     "gmbar_im", 
                     "gbar_na3",
                     "gkdrbar_kdr", 
                     "gcabar_cadyn", 
                     "gsAHPbar_sAHP", 
                     "gkabar_kap",
                     "ghdbar_hd",
                     "glbar_leak"]
)

# Set simulations
simulator = ACTSimulator(output_folder_name = "output")


for sim_idx, amp_value in enumerate([0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.1, 0.3, 0.5, 0.7, 1.0, 3.0, 5.0, 7.0, 9.0]):
    sim_params = SimulationParameters(
        sim_name = "laa_orig_after",
        sim_idx = sim_idx,
        h_v_init=-70,
        h_celsius = 6.3,
        h_dt = 0.1,
        h_tstop = 1000,
        CI = [ConstantCurrentInjection(amp = amp_value, dur = 700, delay = 100)])
    
    train_cell.set_g_bar(["gbar_nap",
                        "gmbar_im", 
                        "gbar_na3",
                        "gkdrbar_kdr", 
                        "gcabar_cadyn", 
                        "gsAHPbar_sAHP", 
                        "gkabar_kap",
                        "ghdbar_hd",
                        "glbar_leak"], orig_g)


    simulator.submit_job(train_cell, sim_params)

simulator.run_jobs(3)

# Combine simulated traces into one dataset for convenience
dp.combine_data("output/laa_orig_after")

simulated_data = np.load("output/laa_orig_after/combined_out.npy")

f = []
for trace_id in range(len(simulated_data)):
    f.append(len(dp.find_events(simulated_data[trace_id, ::10, 0].flatten())))

plt.plot([0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.1, 0.3, 0.5, 0.7, 1.0, 3.0, 5.0, 7.0, 9.0], f)
plt.xlabel("CI (nA)")
plt.ylabel("Frequency (Hz)")
plt.title("FI Curve")
plt.savefig("output/FI_After_Tuning.png")

