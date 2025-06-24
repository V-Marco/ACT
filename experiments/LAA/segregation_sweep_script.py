# Set the path
import sys
sys.path.append("../../")

from act.cell_model import ACTCellModel
from act.simulator import ACTSimulator
from act.types import SimulationParameters, ConstantCurrentInjection, RampCurrentInjection
import act.data_processing as dp

import numpy as np
import pandas as pd

from act.metrics import summary_features_error
import shutil, os

def modify_template(passive_th, lto_th, spiking_th, bursting_th):
    
    passive = [
        ("h.mod", 78, passive_th) # modfile, line
    ]
    lto = [
        ("nap.mod", 65, lto_th), 
        ("im.mod", 73, lto_th)
    ]
    spiking = [
        ("na3.mod", 96, spiking_th), 
        ("kdrca1.mod", 82, spiking_th)
    ]
    bursting = [
        ("kaprox.mod", 110, bursting_th), 
        ("cadyn.mod", 57, bursting_th)
    ]

    for entry in passive + lto + spiking + bursting:
        with open(f"../../data/LAA/orig_cutoff_test/modfiles/{entry[0]}", "r") as file:
            mod_file = file.readlines()

            sign = '>' if entry in passive else '<'
            mod_file[entry[1]] = "\t" + f"if (v {sign} {entry[2]}) " + "{\n"

        with open(f"../../data/LAA/orig_cutoff_test/modfiles/{entry[0]}", "w") as file:
            file.writelines(mod_file)

def run_eval_single_simulation(current_injection, summary_features_target):

    print(os.getcwd())

    # Define the cell
    cell = ACTCellModel(
        cell_name = "Cell_A",
        path_to_hoc_file = "../../data/LAA/orig_cutoff_test/template.hoc",
        path_to_mod_files = "../../data/LAA/orig_cutoff_test/modfiles/",
        passive = ["glbar_leak", "el_leak", "ghdbar_hd"],
        active_channels = ["gbar_na3", "gbar_nap", "gkdrbar_kdr", "gmbar_im", "gkabar_kap", "gcabar_cadyn"],
    )

    # Set simulations
    simulator = ACTSimulator(output_folder_name = "output")
    
    for inj_id, inj in enumerate(current_injection):
        sim_params = SimulationParameters(
            sim_name = "sim",
            sim_idx = inj_id,
            h_celsius = 37,
            h_dt = 0.1,
            h_tstop = 1000,
            h_v_init = -40,
            CI = [inj])

        simulator.submit_job(cell, sim_params)
        
    simulator.run_jobs(3)
    dp.combine_data("output/sim")

    # Get summary features
    data = np.load(f"output/sim/combined_out.npy")
    V = data[:, ::10, 0].reshape((6, -1))
    I = data[:, ::10, 1].reshape((6, -1))
    summary_features_constant = dp.get_summary_features(V[:3], I[:3], window = (100, 800))
    summary_features_ramp = dp.get_summary_features(V[3:], I[3:], window = (400, 800))
    summary_features = pd.concat([summary_features_constant, summary_features_ramp], axis = 0).reset_index(drop = True)

    # Clean
    shutil.rmtree("output/sim")

    return np.nanmean(summary_features_error(summary_features_target.to_numpy(), summary_features.to_numpy()))


if __name__ == "__main__":

    # Get target summary features
    summary_features_target = []
    for sf_type in ["target_constant", "target_ramp"]:
        data = np.load(f"output/{sf_type}/combined_out.npy")
        V = data[:, ::10, 0].reshape((3, -1))
        I = data[:, ::10, 1].reshape((3, -1))
        summary_features_target.append(dp.get_summary_features(V, I, window = (100, 800) if sf_type == "target_constant" else (400, 800)))
    summary_features_target = pd.concat(summary_features_target, axis = 0).reset_index(drop = True)

    all_errors = np.zeros(
    (
        len(np.arange(-80, -50, 5)), # passive
        len(np.arange(-80, -50, 5)), # LTO
        len(np.arange(-80, 10, 5)), # Spiking
        len(np.arange(-80, 10, 5)) # Bursting
    )
    ) * np.nan

    print(f"Shape = {all_errors.shape}")

    for idx_1, passive_bound in enumerate(np.arange(-80, -50, 5)):
        for idx_2, lto_bound in enumerate(np.arange(-80, -50, 5)):
            for idx_3, spiking_bound in enumerate(np.arange(-80, 10, 5)):
                for idx_4, bursting_bound in enumerate(np.arange(-80, 10, 5)):
                    # Set the bounds
                    modify_template(passive_th = passive_bound, lto_th = lto_bound, spiking_th = spiking_bound, bursting_th = bursting_bound)
                    
                    # Run a simulation with constant current injection
                    error = run_eval_single_simulation(
                        [
                            ConstantCurrentInjection(amp = 0, dur = 700, delay = 100),
                            ConstantCurrentInjection(amp = 0.5, dur = 700, delay = 100),
                            ConstantCurrentInjection(amp = 1.0, dur = 700, delay = 100),
                            RampCurrentInjection(amp_incr = 0.01, num_steps = 5, dur = 300, delay = 100, final_step_add_time = 400),
                            RampCurrentInjection(amp_incr = 0.01, num_steps = 10, dur = 300, delay = 100, final_step_add_time = 400),
                            RampCurrentInjection(amp_incr = 0.01, num_steps = 15, dur = 300, delay = 100, final_step_add_time = 400)
                        ],
                        summary_features_target
                    )
                
                    all_errors[idx_1, idx_2, idx_3, idx_4] = error
    
    # Save
    np.save("all_errors.npy", all_errors, allow_pickle = True)