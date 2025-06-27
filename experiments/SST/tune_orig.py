import sys
sys.path.append("../../")
from cell_builder import sst_orig_cell_builder
from act.module import ACTModule

from act.cell_model import ACTCellModel
from act.simulator import ACTSimulator
from act.types import SimulationParameters, ConstantCurrentInjection, OptimizationParameters, ConductanceOptions, FilterParameters

if __name__ == "__main__":
    cell = ACTCellModel(
        cell_name = None,
        path_to_hoc_file = None,
        path_to_mod_files = "../../data/SST/orig/modfiles/",
        passive = ["g_pas", "e_pas", "gbar_Ih"],
        active_channels = ["g_pas", "gbar_Ih", "gbar_Nap", "gbar_Im_v2", "gbar_K_T", "gbar_NaTa", "gbar_Kd", "gbar_Ca_LVA", "gbar_Ca_HVA", "gbar_Kv2like", "gbar_Kv3_1"]
    )
    cell.set_custom_cell_builder(sst_orig_cell_builder)

    sim_params = SimulationParameters(
            sim_name = "cell",
            sim_idx = 0,
            h_celsius = 37,
            h_dt = 0.1,
            h_tstop = 1000
    )

    optim_params = OptimizationParameters(
        conductance_options = [
            ConductanceOptions(variable_name = "g_pas", low = 1e-5, high = 1e-4, n_slices = 5),
            ConductanceOptions(variable_name = "gbar_Ih", low = 1e-4, high = 1e-3, n_slices = 5),
            ConductanceOptions(variable_name = "gbar_Nap", low = 1e-5, high = 1e-3, n_slices = 5),
            ConductanceOptions(variable_name = "gbar_Im_v2", low = 1e-5, high = 1e-3, n_slices = 5),
            ConductanceOptions(variable_name = "gbar_NaTa", low = 1e-3, high = 1e-1, n_slices = 5),
            ConductanceOptions(variable_name = "gbar_Kd", low = 1e-5, high = 1e-2, n_slices = 5),
            ConductanceOptions(variable_name = "gbar_Ca_LVA", low = 1e-4, high = 1e-2, n_slices = 5),
            ConductanceOptions(variable_name = "gbar_Ca_HVA", low = 1e-4, high = 1e-2, n_slices = 5),
            ConductanceOptions(variable_name = "gbar_Kv2like", low = 1e-4, high = 1e-2, n_slices = 5),
            ConductanceOptions(variable_name = "gbar_Kv3_1", low = 1e-4, high = 1e-2, n_slices = 5) 
        ],
        CI_options = [
            ConstantCurrentInjection(amp = 0.2, dur = 700, delay = 100),
            ConstantCurrentInjection(amp = 0.25, dur = 700, delay = 100),
            ConstantCurrentInjection(amp = 0.3, dur = 700, delay = 100),
            ConstantCurrentInjection(amp = 0.35, dur = 700, delay = 100)
        ],
        filter_parameters = FilterParameters(
            saturation_threshold = -55,
            window_of_inspection = (100, 800)
        ),
        n_cpus = 2
    )

    m = ACTModule(
        name = "orig",
        cell = cell,
        simulation_parameters = sim_params,
        optimization_parameters = optim_params,
        target_file = "/Users/vladimiromelyusik/PV_Cell/target_sf.csv"
    )
    m.run()