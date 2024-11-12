

def run():
    num_seeds = 5
    for seed in range(42,42+num_seeds): 
        from act.SyntheticGenerator import SyntheticGenerator
        from act.act_types import SimParams, OptimizationParameters
        from act.cell_model import TargetCell, ModuleParameters

        random_seed = seed
        num_slices = 10
        experiment_folder = f"output/BursterIzh_Original_sl-{num_slices}/{random_seed}"
        target_folder = experiment_folder + "/target"

        module_folder = experiment_folder + "/module"

        target_cell = TargetCell(
            hoc_file="/home/mwsrgf/proj/ACT/data/Burster_Izhikevich/orig/template.hoc",
            mod_folder="/home/mwsrgf/proj/ACT/data/Burster_Izhikevich/orig",
            cell_name="Burster_Izh",
            g_names = ["gbar_na3", "gkdrbar_kdr","gbar_nap","gmbar_im", "glbar_leak"]
        )

        sim_par= SimParams(
                    h_v_init=-67.0,  # Consistent with nrngui settings
                    h_tstop=500,     # Simulation time in ms
                    h_dt=0.1,      # Time step in ms
                    h_celsius=6.3,   # Temperature in degrees Celsius
                    CI_type="constant",
                    CI_amps=[0.1,0.3,0.5],   # Current injection amplitude
                    CI_dur=300,      # Duration of current injection
                    CI_delay=100,     # Delay before current injection
                    set_g_to=[]
                )

        sg = SyntheticGenerator(
            ModuleParameters(
                module_folder_name=target_folder,
                cell= target_cell,
                sim_params= sim_par,
                optim_params = OptimizationParameters(
                    blocked_channels= []
                )
            )
        )

        sg.generate_synthetic_target_data("target_data.csv")

        from act.cell_model import TrainCell

        train_cell = TrainCell(
            hoc_file="/home/mwsrgf/proj/ACT/data/Burster_Izhikevich/orig/template.hoc",
            mod_folder="/home/mwsrgf/proj/ACT/data/Burster_Izhikevich/orig",
            cell_name="Burster_Izh",
            g_names = ["gbar_na3", "gkdrbar_kdr","gbar_nap","gmbar_im", "glbar_leak"]
        )

        from act.ACTModule import ACTModule
        from act.act_types import OptimizationParameters, OptimizationParam
        from act.cell_model import ModuleParameters


        mod = ACTModule(
            ModuleParameters(
                module_folder_name=module_folder,
                cell= train_cell,
                target_traces_file = f"{target_folder}/target_data.csv",
                sim_params= sim_par,
                optim_params= OptimizationParameters(
                    g_ranges_slices= [
                        OptimizationParam(param="gbar_na3", low=0.015, high=0.045, n_slices=num_slices),
                        OptimizationParam(param="gkdrbar_kdr", low=0.019, high=0.047, n_slices=num_slices),
                        OptimizationParam(param="gbar_nap", low=0.00015, high=0.00045, n_slices=num_slices),
                        OptimizationParam(param="gmbar_im", low=0.00165, high=0.00495, n_slices=num_slices),
                        OptimizationParam(param="glbar_leak", low=0.0, high=0.005, n_slices=num_slices)
                    ],
                    #filtered_out_features = ["no_spikes", "saturated"],
                    train_features=["i_trace_stats", "number_of_spikes", "spike_times", "spike_height_stats", "trough_times", "trough_height_stats"],
                    spike_threshold=0,
                    saturation_threshold=-55,
                    first_n_spikes=20,
                    prediction_eval_method='features',
                    random_state=random_seed,
                    save_file=f"{module_folder}/results/saved_metrics.json"
                )
            )
        )
        final_predicted_g_data_file = mod.run()
        mod.pickle_rf(mod.rf_model,f"{module_folder}/trained_rf.pkl")
        print(train_cell.predicted_g)

        from act import ACTPlot
        ACTPlot.plot_v_comparison(
            final_predicted_g_data_file, 
            module_folder, 
            sim_par["CI_amps"],
            sim_par["h_dt"]
            )

        ACTPlot.plot_fi_comparison(
            module_folder, 
            sim_par["CI_amps"]
            )

        from act.Metrics import Metrics

        metrics = Metrics()

        mean, stdev = metrics.save_interspike_interval_comparison(
            module_folder,
            final_predicted_g_data_file,
            sim_par["CI_amps"], 
            sim_par["h_dt"],
            first_n_spikes=5,
            save_file=f"{module_folder}/results/saved_metrics.json"
        )

        metrics.save_prediction_g_mae(
            actual_g={"gbar_na3": 0.03, "gkdrbar_kdr": 0.028,"gbar_nap": 0.0003,"gmbar_im": 0.0033, "glbar_leak": 3.5e-5},
            save_file=f"{module_folder}/results/saved_metrics.json"
        )

        from act import ACTPlot as actplt

        g_names = ["gbar_na3", "gkdrbar_kdr","gbar_nap","gmbar_im"]

        for i in range(len(g_names)-1):
            actplt.plot_training_feature_mae_contour_plot(
                module_folder,
                sim_par["CI_amps"],
                sim_par["CI_delay"],
                sim_par["h_dt"],
                index1=0,
                index2=i+1,
                g_names=g_names,
                train_features=["i_trace_stats", "number_of_spikes", "spike_times", "spike_height_stats", "trough_times", "trough_height_stats"],
                threshold=0,
                first_n_spikes=20,
                num_levels=100,
                results_filename=f"{module_folder}/results/Feature_MAE_Contour_Plot_{g_names[0]}_{g_names[i+1]}.png"
            )
            
        from act import ACTPlot as actplt

        g_names = ["gbar_na3", "gkdrbar_kdr","gbar_nap","gmbar_im"]

        for i in range(len(g_names)-1):
            actplt.plot_training_fi_mae_contour_plot(
                module_folder,
                sim_par["CI_amps"], 
                sim_par["CI_dur"],
                sim_par["CI_delay"],
                sim_par["h_dt"],
                index1=0,
                index2=i+1,
                g_names=g_names,
                results_filename=f"{module_folder}/results/FI_MAE_Contour_Plot_{g_names[0]}_{g_names[i+1]}.png"
            )
            
        from act import ACTPlot as actplt

        g_names = ["gbar_na3", "gkdrbar_kdr","gbar_nap","gmbar_im"]

        for i in range(len(g_names)-1):
            actplt.plot_training_v_mae_contour_plot(
                module_folder,
                sim_par["CI_amps"], 
                sim_par["CI_delay"],
                sim_par["h_dt"],
                index1=0,
                index2=i+1,
                g_names=g_names,
                results_filename=f"{module_folder}/results/Voltage_MAE_Contour_Plot_{g_names[0]}_{g_names[i+1]}.png"
            )
        
        
if __name__ == "__main__":
    run()