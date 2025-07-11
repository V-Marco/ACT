import sys
sys.path.append("../../")
from cell_builder import L5_seg_cell_builder
from act.module import ACTModule

from act.cell_model import ACTCellModel
from act.types import SimulationParameters, ConstantCurrentInjection, OptimizationParameters, ConductanceOptions, FilterParameters
import pickle

if __name__ == "__main__":

    train_cell = ACTCellModel(
        cell_name = None,
        path_to_hoc_file = None,
        path_to_mod_files = "../../data/L5/seg_tuned/modfiles/",
        passive = ["g_pas", "e_pas", "gbar_Ih"],
        active_channels = ["gbar_Nap", "gbar_K_T", "gbar_NaTa", "gbar_Kd", "gbar_Ca_LVA", "gbar_Ca_HVA", "gbar_Kv2like", "gbar_Kv3_1"]
    )
    train_cell.set_custom_cell_builder(L5_seg_cell_builder)

    # Module 3: Bursting
    sim_params = SimulationParameters(
            sim_name = "cell",
            sim_idx = 0,
            h_celsius = 37,
            h_dt = 0.1,
            h_tstop = 1000
    )

    optim_params = OptimizationParameters(
        conductance_options = [
            ConductanceOptions(variable_name = "gbar_Ca_LVA", low = 0.0008, high = 0.008, n_slices = 5),
            ConductanceOptions(variable_name = "gbar_Ca_HVA", low = 0.001, high = 0.01, n_slices = 5),
            ConductanceOptions(variable_name = "gbar_Kv2like", low = 0.001, high = 0.02, n_slices = 5),
            ConductanceOptions(variable_name = "gbar_Kv3_1", low = 0.01, high = 0.09, n_slices = 5)
        ],
        CI_options = [
            ConstantCurrentInjection(amp = 0.15, dur = 700, delay = 100),
            ConstantCurrentInjection(amp = 0.19, dur = 700, delay = 100),
            ConstantCurrentInjection(amp = 0.23, dur = 700, delay = 100),
            ConstantCurrentInjection(amp = 0.27, dur = 700, delay = 100),
            ConstantCurrentInjection(amp = 0.33, dur = 700, delay = 100)
        ],
        filter_parameters = FilterParameters(
            saturation_threshold = -55,
            window_of_inspection = (100, 800)
        ),
        n_cpus = 20
    )

    m = ACTModule(
        name = "seg_bursting",
        cell = train_cell,
        simulation_parameters = sim_params,
        optimization_parameters = optim_params,
        target_file = "target_sf.csv"
    )
    metrics = m.run()
    metrics.to_csv("metrics_bursting.csv", index = False)
    with open("bursting_pred.pickle", "wb") as file:
        pickle.dump(m.cell.prediction, file)

