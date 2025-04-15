
def run():
    num_seeds = 5
    for seed in range(42,42+num_seeds): 
        from act.SyntheticGenerator import SyntheticGenerator
        from act.act_types import SimParams, OptimizationParameters
        from act.cell_model import TargetCell, ModuleParameters

        random_seed = seed
        num_slices_new = 10
        num_slices_old = 5
        experiment_folder = f"output/Spiker_Full_vhalf_vcutoff_blocked_fsl-{num_slices_new}_{num_slices_old}/{random_seed}"
        target_folder = experiment_folder + "/target"

        # module 1 is for spiking, final is a refinement of all previous modules (except passive props)
        module_1_folder = experiment_folder + "/module_1"
        module_2_folder = experiment_folder + "/module_2"
        module_final_folder = experiment_folder + "/module_final"

        target_cell = TargetCell(
            hoc_file="/home/mwsrgf/proj/ACT/data/Spiker_Izhikevich/seg/template.hoc",
            mod_folder="/home/mwsrgf/proj/ACT/data/Spiker_Izhikevich/seg",
            cell_name="Spiker_Izhikevich_seg",
            g_names = ["gnabar_hh_seg", "gkbar_hh_seg", "gl_hh_seg"]
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

        passive_sim_par = SimParams(
                    h_v_init = -65,
                    h_tstop = 1500,
                    h_dt = 0.001,
                    h_celsius = 6.3,
                    CI_type = "constant",
                    CI_amps = [-0.1],    ##################### NEGATIVE CURRENT INJECTION ###################
                    CI_dur = 1000,
                    CI_delay = 500,
                    set_g_to=[]
                )

        sg_passive = SyntheticGenerator(
            ModuleParameters(
                module_folder_name=target_folder,
                cell= target_cell,
                sim_params= passive_sim_par,
                optim_params = OptimizationParameters(
                    blocked_channels= []
                )
            )
        )

        sg_passive.generate_synthetic_target_data("passive_data.csv")

        from act.PassivePropertiesModule import PassivePropertiesModule
        from act.act_types import SimParams
        from act.cell_model import TrainCell

        train_cell = TrainCell(
            hoc_file="/home/mwsrgf/proj/ACT/data/Spiker_Izhikevich/seg/template.hoc",
            mod_folder="/home/mwsrgf/proj/ACT/data/Spiker_Izhikevich/seg",
            cell_name="Spiker_Izhikevich_seg",
            g_names = ["gnabar_hh_seg", "gkbar_hh_seg", "gl_hh_seg"]
        )

        passive_mod = PassivePropertiesModule(
            train_cell=train_cell,
            sim_params=passive_sim_par,
            trace_filepath=f"{target_folder}/passive_data.csv",
            leak_conductance_variable="gl_hh_seg",
            leak_reversal_variable="el_hh_seg"
        )

        passive_mod.set_passive_properties()
        print(train_cell.passive_properties)

        from act.ACTModule import ACTModule
        from act.act_types import OptimizationParameters, OptimizationParam
        from act.cell_model import ModuleParameters

        mod1 = ACTModule(
            ModuleParameters(
                module_folder_name=module_1_folder,
                cell= train_cell,
                target_traces_file = f"{target_folder}/target_data.csv",
                sim_params= sim_par,
                optim_params= OptimizationParameters(
                g_ranges_slices= [
                    OptimizationParam(param="gnabar_hh_seg", low=0.06, high=0.18, blocked=True, n_slices=1),
                    OptimizationParam(param="gkbar_hh_seg", low=0.018, high=0.054, n_slices=num_slices_new),
                    OptimizationParam(param="gl_hh_seg", prediction=train_cell.passive_properties.g_bar_leak, bounds_variation=0.0, n_slices=1)
                ],
                    filtered_out_features = ["no_spikes", "saturated"],
                    train_features=["i_trace_stats", "number_of_spikes", "spike_times", "spike_height_stats", "number_of_troughs", "trough_times", "trough_height_stats"],
                    prediction_eval_method='fi_curve',
                    spike_threshold=0,
                    saturation_threshold=-65,
                    first_n_spikes=20,
                    random_state=random_seed,
                    save_file=f"{module_1_folder}/results/saved_metrics.json"
                )
            )
        )

        predicted_g_data_file = mod1.run()
        mod1.pickle_rf(mod1.rf_model,f"{module_1_folder}/trained_rf.pkl")

        from act.ACTModule import ACTModule
        from act.act_types import OptimizationParameters, OptimizationParam
        from act.cell_model import ModuleParameters

        bounds_variation = 0.15

        mod2 = ACTModule(
            ModuleParameters(
                module_folder_name=module_2_folder,
                cell= train_cell,
                target_traces_file = f"{target_folder}/target_data.csv",
                sim_params= sim_par,
                optim_params= OptimizationParameters(
                g_ranges_slices= [
                    OptimizationParam(param="gnabar_hh_seg",  low=0.06, high=0.18, n_slices=num_slices_new),
                    OptimizationParam(param="gkbar_hh_seg", prediction=train_cell.predicted_g["gkbar_hh_seg"], bounds_variation=train_cell.predicted_g["gkbar_hh_seg"] * bounds_variation, n_slices=num_slices_old),
                    OptimizationParam(param="gl_hh_seg", prediction=train_cell.passive_properties.g_bar_leak, bounds_variation=0.0, n_slices=1)
                ],
                    filtered_out_features = ["no_spikes", "saturated"],
                    train_features=["i_trace_stats", "number_of_spikes", "spike_times", "spike_height_stats", "number_of_troughs", "trough_times", "trough_height_stats"],
                    prediction_eval_method='fi_curve',
                    spike_threshold=0,
                    saturation_threshold=-65,
                    first_n_spikes=20,
                    random_state=random_seed,
                    save_file=f"{module_2_folder}/results/saved_metrics.json"
                )
            )
        )

        predicted_g_data_file = mod2.run()
        mod2.pickle_rf(mod2.rf_model,f"{module_2_folder}/trained_rf.pkl")

        from act.ACTModule import ACTModule
        from act.act_types import OptimizationParameters, OptimizationParam
        from act.cell_model import ModuleParameters

        bounds_variation = 0.15


        final_mod = ACTModule(
            ModuleParameters(
                module_folder_name=module_final_folder,
                cell= train_cell,
                target_traces_file = f"{target_folder}/target_data.csv",
                sim_params= sim_par,
                optim_params= OptimizationParameters(
                    g_ranges_slices= [
                        OptimizationParam(param="gnabar_hh_seg", prediction=train_cell.predicted_g["gnabar_hh_seg"],bounds_variation=train_cell.predicted_g["gnabar_hh_seg"] * bounds_variation, n_slices=num_slices_old),
                        OptimizationParam(param="gkbar_hh_seg", prediction=train_cell.predicted_g["gkbar_hh_seg"], bounds_variation=train_cell.predicted_g["gkbar_hh_seg"] * bounds_variation, n_slices=num_slices_old),
                        OptimizationParam(param="gl_hh_seg", prediction=train_cell.passive_properties.g_bar_leak, bounds_variation=0.0, n_slices=1)
                    ],
                    filtered_out_features = ["no_spikes", "saturated"],
                    train_features=["i_trace_stats", "number_of_spikes", "spike_times", "spike_height_stats", "number_of_troughs", "trough_times", "trough_height_stats"],
                    spike_threshold=0,
                    saturation_threshold=-65,
                    first_n_spikes=20,
                    prediction_eval_method='fi_curve',
                    random_state=random_seed,
                    previous_modules=["module_1","module_2"],
                    save_file=f"{module_final_folder}/results/saved_metrics.json"
                )
            )
        )

        final_predicted_g_data_file = final_mod.run()
        final_mod.pickle_rf(final_mod.rf_model,f"{module_final_folder}/trained_rf.pkl")

        from act import act_plot
        act_plot.plot_v_comparison(
            final_predicted_g_data_file, 
            module_final_folder, 
            sim_par["CI_amps"],
            sim_par["h_dt"]
            )

        act_plot.plot_fi_comparison(
            module_final_folder, 
            sim_par["CI_amps"]
            )

        from act.metrics import Metrics

        metrics = Metrics()

        mean, stdev = metrics.save_interspike_interval_comparison(
            module_final_folder,
            final_predicted_g_data_file,
            sim_par["CI_amps"], 
            sim_par["h_dt"],
            first_n_spikes=5,
            save_file=f"{module_final_folder}/results/saved_metrics.json"
        )


        metrics.save_prediction_g_mae(
            actual_g={'gnabar_hh_seg': 0.12, 'gkbar_hh_seg': 0.036, 'gl_hh_seg': 0.00046907},
            save_file=f"{module_final_folder}/results/saved_metrics.json"
        )

        metrics.save_feature_mae(
            module_final_folder,
            final_predicted_g_data_file,
            ["i_trace_stats", "number_of_spikes", "spike_times", "spike_height_stats", "number_of_troughs", "trough_times", "trough_height_stats"],
            sim_par["h_dt"],
            first_n_spikes=5,
            save_file=f"{module_final_folder}/results/saved_metrics.json"
        )


        from act import act_plot as actplt

        g_names = ["gkbar_hh_seg", "gl_hh_seg"]

        actplt.plot_training_fi_mae_contour_plot(
        module_final_folder,
        sim_par["CI_amps"],
        sim_par["CI_dur"],
        sim_par["CI_delay"],
        sim_par["h_dt"],
        index1=0,
        index2=1,
        g_names=g_names,
        num_levels=100,
        results_filename=f"{module_final_folder}/results/Fi_MAE_Contour_Plot.png"
        )

        from act import act_plot as actplt

        g_names = ["gkbar_hh_seg", "gl_hh_seg"]

        actplt.plot_training_v_mae_contour_plot(
        module_final_folder,
        sim_par["CI_amps"],
        sim_par["CI_delay"],
        sim_par["h_dt"],
        index1=0,
        index2=1,
        g_names=g_names,
        num_levels=100,
        results_filename=f"{module_final_folder}/results/V_Trace_Contour_Plot.png"
        )

        from act import act_plot as actplt

        g_names = ["gkbar_hh_seg", "gl_hh_seg"]


        actplt.plot_training_feature_mae_contour_plot(
            module_final_folder,
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
            results_filename=f"{module_final_folder}/results/Feature_MAE_Contour_Plot_{g_names[0]}_{g_names[1]}.png"
        )
    
if __name__ == "__main__":
        run()