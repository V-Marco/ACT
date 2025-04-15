
def run():
    num_seeds = 5
    for seed in range(42,42+num_seeds): 
        from act.SyntheticGenerator import SyntheticGenerator
        from act.act_types import SimParams, OptimizationParameters
        from act.cell_model import TargetCell, ModuleParameters

        random_seed = seed
        num_slices = 10

        experiment_folder = f"output/Spiker_Original_fsl-{num_slices}/{random_seed}"
        target_folder = experiment_folder + "/target"
        final_folder = experiment_folder + "/final"

        target_cell = TargetCell(
            hoc_file="/home/mwsrgf/proj/ACT/data/Spiker_Izhikevich/orig/template.hoc",
            mod_folder="/home/mwsrgf/proj/ACT/data/Spiker_Izhikevich/orig",
            cell_name="Spiker_Izhikevich_orig",
            g_names = ["gnabar_hh_orig", "gkbar_hh_orig", "gl_hh_orig"]
        )

        sim_par= SimParams(
                    h_v_init=-65.0,  # Consistent with nrngui settings
                    h_tstop=500,     # Simulation time in ms
                    h_dt=0.1,      # Time step in ms
                    h_celsius=6.3,   # Temperature in degrees Celsius
                    CI_type="constant",
                    CI_amps=[0.1,0.2,0.3],   # Current injection amplitude
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

        from act.ACTModule import ACTModule
        from act.SyntheticGenerator import SyntheticGenerator
        from act.act_types import  Cell, SimParams, OptimizationParameters, OptimizationParam
        from act.cell_model import ModuleParameters
        from act.PassivePropertiesModule import PassivePropertiesModule
        from act.act_types import SimParams
        from act.cell_model import TrainCell

        train_cell = TrainCell(
            hoc_file="/home/mwsrgf/proj/ACT/data/Spiker_Izhikevich/orig/template.hoc",
            mod_folder="/home/mwsrgf/proj/ACT/data/Spiker_Izhikevich/orig",
            cell_name="Spiker_Izhikevich_orig",
            g_names = ["gnabar_hh_orig", "gkbar_hh_orig", "gl_hh_orig"]
        )

        mod = ACTModule(
            ModuleParameters(
                module_folder_name=final_folder,
                cell= train_cell,
                target_traces_file = f"{target_folder}/target_data.csv",
                sim_params= sim_par,
                optim_params= OptimizationParameters(
                    g_ranges_slices= [
                        OptimizationParam(param="gnabar_hh_seg", low=0.0, high=0.3, n_slices=num_slices),
                        OptimizationParam(param="gkbar_hh_seg", low=0.0, high=0.3, n_slices=num_slices),
                        OptimizationParam(param="gl_hh_seg", low=0.0, high=0.3, n_slices=num_slices),
                        
                    ],
                    filtered_out_features = ["no_spikes", "saturated"],
                    train_features=["i_trace_stats", "number_of_spikes", "spike_times", "spike_height_stats", "number_of_troughs", "trough_times", "trough_height_stats"],
                    prediction_eval_method='fi_curve',
                    spike_threshold=0,
                    saturation_threshold=-55,
                    window_of_inspection=(120,400),
                    first_n_spikes=20,
                    random_state=random_seed,
                    save_file=f"{final_folder}/results/saved_metrics.json"
                )
            )
        )

        predicted_g_data_file = mod.run()
        mod.pickle_rf(mod.rf_model,f"{final_folder}/trained_rf.pkl")

        from act import act_plot
        act_plot.plot_v_comparison(
            predicted_g_data_file, 
            final_folder, 
            sim_par["CI_amps"],
            sim_par["h_dt"]
            )

        act_plot.plot_fi_comparison(
            final_folder, 
            sim_par["CI_amps"]
            )

        from act.metrics import Metrics

        metrics = Metrics()

        mean, stdev = metrics.save_interspike_interval_comparison(
            final_folder,
            predicted_g_data_file,
            sim_par["CI_amps"], 
            sim_par["h_dt"],
            first_n_spikes=5,
            save_file=f"{final_folder}/results/saved_metrics.json"
        )


        metrics.save_prediction_g_mae(
            actual_g={'gnabar_hh_orig': 0.12, 'gkbar_hh_orig': 0.036, 'gl_hh_orig': 0.0003},
            save_file=f"{final_folder}/results/saved_metrics.json"
        )

        metrics.save_feature_mae(
            final_folder,
            predicted_g_data_file,
            ["i_trace_stats", "number_of_spikes", "spike_times", "spike_height_stats", "number_of_troughs", "trough_times", "trough_height_stats"],
            sim_par["h_dt"],
            first_n_spikes=5,
            save_file=f"{final_folder}/results/saved_metrics.json"
        )

        from act import act_plot as actplt

        g_names = ["gkbar_hh_orig", "gl_hh_orig"]

        actplt.plot_training_fi_mae_contour_plot(
        final_folder,
        sim_par["CI_amps"],
        sim_par["CI_dur"],
        sim_par["CI_delay"],
        sim_par["h_dt"],
        index1=0,
        index2=1,
        g_names=g_names,
        num_levels=100,
        results_filename=f"{final_folder}/results/Fi_MAE_Contour_Plot.png"
        )

        from act import act_plot as actplt

        g_names = ["gkbar_hh_orig", "gl_hh_orig"]

        actplt.plot_training_v_mae_contour_plot(
        final_folder,
        sim_par["CI_amps"],
        sim_par["CI_delay"],
        sim_par["h_dt"],
        index1=0,
        index2=1,
        g_names=g_names,
        num_levels=100,
        results_filename=f"{final_folder}/results/V_Trace_Contour_Plot.png"
        )

        from act import act_plot as actplt

        g_names = ["gkbar_hh_orig", "gl_hh_orig"]


        actplt.plot_training_feature_mae_contour_plot(
            final_folder,
            sim_par["CI_amps"],
            sim_par["CI_delay"],
            sim_par["h_dt"],
            index1=0,
            index2=1,
            g_names=g_names,
            train_features=["i_trace_stats", "number_of_spikes", "spike_times", "spike_height_stats", "number_of_troughs", "trough_times", "trough_height_stats"],
            threshold=0,
            first_n_spikes=20,
            num_levels=100,
            results_filename=f"{final_folder}/results/Feature_MAE_Contour_Plot_{g_names[0]}_{g_names[1]}.png"
        )
        
if __name__ == "__main__":
        run()