
def run():
    num_seeds = 5
    for seed in range(42,42+num_seeds): 
        from act.SyntheticGenerator import SyntheticGenerator
        from act.act_types import SimParams, OptimizationParameters
        from act.cell_model import TargetCell, ModuleParameters

        random_seed = seed
        num_slices_new = 10
        num_slices_old = 5
        # num of expected jobs: 6555
        cell_props = "low"   # low, high, center
        experiment = "range" # blocked, range
        experiment_folder = f"output/BursterIzh_Full_vhalf_vcutoff_fsl-{num_slices_new}-{num_slices_old}/{random_seed}"
        #experiment_folder = f"output/bursterIzh_seg_surfaceTest_sl-{num_slices_new}/{random_seed}"
        target_folder = experiment_folder + "/target"

        # module 1 is for spiking, module 2 for bursting, final for refining all channels

        module_1_folder = experiment_folder + "/module_1"
        module_2_folder = experiment_folder + "/module_2"
        module_3_folder = experiment_folder + "/module_3"
        module_4_folder = experiment_folder + "/module_4"
        module_final_folder = experiment_folder + "/module_final"

        # We need to order the ion channels according to V1/2:
        # im: -52.7
        # kdr: 13
        # na3: -30 (act), -45 (inact)
        # nap: -48
        # In order: leak, im, nap, na3, kdr

        target_cell = TargetCell(
            hoc_file=f"/home/mwsrgf/proj/ACT/data/Burster_Izhikevich/seg/template.hoc",
            mod_folder=f"/home/mwsrgf/proj/ACT/data/Burster_Izhikevich/seg",
            cell_name="Burster_Izh",
            g_names = ["gmbar_im", "gbar_nap", "gbar_na3", "gkdrbar_kdr", "glbar_leak"]
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

        passive_sim_par = SimParams(
                    h_v_init = -67,
                    h_tstop = 1500,
                    h_dt = 0.001,
                    h_celsius = 6.3,
                    CI_type = "constant",
                    CI_amps = [-1],    ##################### NEGATIVE CURRENT INJECTION ###################
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
            hoc_file=f"/home/mwsrgf/proj/ACT/data/Burster_Izhikevich/seg/template.hoc",
            mod_folder=f"/home/mwsrgf/proj/ACT/data/Burster_Izhikevich/seg",
            cell_name="Burster_Izh",
            g_names = ["gmbar_im", "gbar_nap", "gbar_na3", "gkdrbar_kdr", "glbar_leak"]
        )

        passive_mod = PassivePropertiesModule(
            train_cell=train_cell,
            sim_params=passive_sim_par,
            trace_filepath=f"{target_folder}/passive_data.csv",
            leak_conductance_variable="glbar_leak",
            leak_reversal_variable="el_leak"
        )

        passive_mod.set_passive_properties()
        print(train_cell.passive_properties)

        from act.ACTModule import ACTModule
        from act.act_types import OptimizationParameters, OptimizationParam
        from act.cell_model import ModuleParameters

        # gmbar_im = 0.0038, gkdrbar_kdr= 0.03, gbar_nap= 0.0004, gbar_na3= 0.05, glbar_leak= 3.6e-5
        mod1 = ACTModule(
            ModuleParameters(
                module_folder_name=module_1_folder,
                cell= train_cell,
                target_traces_file = f"{target_folder}/target_data.csv",
                sim_params= sim_par,
                optim_params= OptimizationParameters(
                    g_ranges_slices= [
                        OptimizationParam(param="gmbar_im", low=0.0019, high=0.0057, n_slices=num_slices_new),
                        OptimizationParam(param="gbar_nap", low=0.0002, high=0.0006, n_slices=1),
                        OptimizationParam(param="gbar_na3", low=0.025, high=0.075, n_slices=1),
                        OptimizationParam(param="gkdrbar_kdr", low=0.015, high=0.045, n_slices=1),
                        OptimizationParam(param="glbar_leak", prediction=train_cell.passive_properties.g_bar_leak, bounds_variation=0.0, n_slices=1)
                    ],
                    filtered_out_features = ["no_spikes", "saturated"],
                    train_features=["i_trace_stats", "number_of_spikes", "spike_times", "spike_height_stats", "trough_times", "trough_height_stats"],
                    prediction_eval_method='features',
                    spike_threshold=0,
                    saturation_threshold=-55,
                    window_of_inspection=(120,400),
                    first_n_spikes=20,
                    random_state=random_seed,
                    save_file=f"{module_1_folder}/results/saved_metrics.json"
                )
            )
        )
        
        predicted_g_data_file = mod1.run()
        mod1.pickle_rf(mod1.rf_model,f"{module_1_folder}/trained_rf.pkl")
        print(train_cell.predicted_g)

        from act.ACTModule import ACTModule
        from act.act_types import OptimizationParameters, OptimizationParam
        from act.cell_model import ModuleParameters

        # Allow all channels to vary by 20% their predicted value in the previous module
        bounds_variation = 0.15

        # gmbar_im = 0.0038, gkdrbar_kdr= 0.03, gbar_nap= 0.0004, gbar_na3= 0.05, glbar_leak= 3.6e-5
        mod2 = ACTModule(
            ModuleParameters(
                module_folder_name=module_2_folder,
                cell= train_cell,
                target_traces_file = f"{target_folder}/target_data.csv",
                sim_params= sim_par,
                optim_params= OptimizationParameters(
                    g_ranges_slices = [
                        OptimizationParam(param="gmbar_im", prediction=train_cell.predicted_g["gmbar_im"], bounds_variation=train_cell.predicted_g["gmbar_im"]*bounds_variation, n_slices=num_slices_old),
                        OptimizationParam(param="gbar_nap", low=0.0002, high=0.0006, n_slices=num_slices_new),
                        OptimizationParam(param="gbar_na3", low=0.025, high=0.075, n_slices=1),
                        OptimizationParam(param="gkdrbar_kdr", low=0.015, high=0.045, n_slices=1),
                        OptimizationParam(param="glbar_leak", prediction=train_cell.passive_properties.g_bar_leak, bounds_variation=0.0, n_slices=1)
                    ],
                    filtered_out_features = ["no_spikes", "saturated"],
                    train_features=["i_trace_stats", "number_of_spikes", "spike_times", "spike_height_stats", "trough_times", "trough_height_stats"],
                    prediction_eval_method='features',
                    spike_threshold=0,
                    saturation_threshold=-55,
                    first_n_spikes=20,
                    random_state=random_seed,
                    save_file=f"{module_2_folder}/results/saved_metrics.json"
                )
            )
        )

        predicted_g_data_file = mod2.run()

        mod2.pickle_rf(mod2.rf_model,f"{module_2_folder}/trained_rf.pkl")
        print(train_cell.predicted_g)

        from act.ACTModule import ACTModule
        from act.act_types import OptimizationParameters, OptimizationParam
        from act.cell_model import ModuleParameters

        # Allow all channels to vary by 20% their predicted value in the previous module
        bounds_variation = 0.15

        # gmbar_im = 0.0038, gkdrbar_kdr= 0.03, gbar_nap= 0.0004, gbar_na3= 0.05, glbar_leak= 3.6e-5
        mod3 = ACTModule(
            ModuleParameters(
                module_folder_name=module_3_folder,
                cell= train_cell,
                target_traces_file = f"{target_folder}/target_data.csv",
                sim_params= sim_par,
                optim_params= OptimizationParameters(
                    g_ranges_slices = [
                        OptimizationParam(param="gmbar_im", prediction=train_cell.predicted_g["gmbar_im"], bounds_variation=train_cell.predicted_g["gmbar_im"]*bounds_variation, n_slices=num_slices_old),
                        OptimizationParam(param="gbar_nap", prediction=train_cell.predicted_g["gbar_nap"], bounds_variation=train_cell.predicted_g["gbar_nap"]*bounds_variation, n_slices=num_slices_old),
                        OptimizationParam(param="gbar_na3", low=0.025, high=0.075, n_slices=num_slices_new),
                        OptimizationParam(param="gkdrbar_kdr", low=0.015, high=0.045, n_slices=1),
                        OptimizationParam(param="glbar_leak", prediction=train_cell.passive_properties.g_bar_leak, bounds_variation=0.0, n_slices=1)
                    ],
                    filtered_out_features = ["no_spikes", "saturated"],
                    train_features=["i_trace_stats", "number_of_spikes", "spike_times", "spike_height_stats", "trough_times", "trough_height_stats"],
                    prediction_eval_method='features',
                    spike_threshold=0,
                    saturation_threshold=-55,
                    first_n_spikes=20,
                    random_state=random_seed,
                    save_file=f"{module_3_folder}/results/saved_metrics.json"
                )
            )
        )

        predicted_g_data_file = mod3.run()

        mod3.pickle_rf(mod3.rf_model,f"{module_3_folder}/trained_rf.pkl")
        print(train_cell.predicted_g)

        from act.ACTModule import ACTModule
        from act.act_types import OptimizationParameters, OptimizationParam
        from act.cell_model import ModuleParameters

        # Allow all channels to vary by 20% their predicted value in the previous module
        bounds_variation = 0.15

        # gmbar_im = 0.0038, gkdrbar_kdr= 0.03, gbar_nap= 0.0004, gbar_na3= 0.05, glbar_leak= 3.6e-5
        mod4 = ACTModule(
            ModuleParameters(
                module_folder_name=module_4_folder,
                cell= train_cell,
                target_traces_file = f"{target_folder}/target_data.csv",
                sim_params= sim_par,
                optim_params= OptimizationParameters(
                    g_ranges_slices = [
                        OptimizationParam(param="gmbar_im", prediction=train_cell.predicted_g["gmbar_im"], bounds_variation=train_cell.predicted_g["gmbar_im"]*bounds_variation, n_slices=num_slices_old),
                        OptimizationParam(param="gbar_nap", prediction=train_cell.predicted_g["gbar_nap"], bounds_variation=train_cell.predicted_g["gbar_nap"]*bounds_variation, n_slices=num_slices_old),
                        OptimizationParam(param="gbar_na3", prediction=train_cell.predicted_g["gbar_na3"], bounds_variation=train_cell.predicted_g["gbar_na3"]*bounds_variation, n_slices=num_slices_old),
                        OptimizationParam(param="gkdrbar_kdr", low=0.015, high=0.045, n_slices=num_slices_new),
                        OptimizationParam(param="glbar_leak", prediction=train_cell.passive_properties.g_bar_leak, bounds_variation=0.0, n_slices=1)
                    ],
                    filtered_out_features = ["no_spikes", "saturated"],
                    train_features=["i_trace_stats", "number_of_spikes", "spike_times", "spike_height_stats", "trough_times", "trough_height_stats"],
                    prediction_eval_method='features',
                    spike_threshold=0,
                    saturation_threshold=-55,
                    first_n_spikes=20,
                    random_state=random_seed,
                    save_file=f"{module_4_folder}/results/saved_metrics.json"
                )
            )
        )

        predicted_g_data_file = mod4.run()
        mod4.pickle_rf(mod4.rf_model,f"{module_4_folder}/trained_rf.pkl")
        print(train_cell.predicted_g)

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
                    g_ranges_slices = [
                        OptimizationParam(param="gmbar_im", prediction=train_cell.predicted_g["gmbar_im"], bounds_variation=train_cell.predicted_g["gmbar_im"]*bounds_variation, n_slices=num_slices_old),
                        OptimizationParam(param="gbar_nap", prediction=train_cell.predicted_g["gbar_nap"], bounds_variation=train_cell.predicted_g["gbar_nap"]*bounds_variation, n_slices=num_slices_old),
                        OptimizationParam(param="gbar_na3", prediction=train_cell.predicted_g["gbar_na3"], bounds_variation=train_cell.predicted_g["gbar_na3"]*bounds_variation, n_slices=num_slices_old),
                        OptimizationParam(param="gkdrbar_kdr", prediction=train_cell.predicted_g["gkdrbar_kdr"], bounds_variation=train_cell.predicted_g["gkdrbar_kdr"]*bounds_variation, n_slices=num_slices_old),
                        OptimizationParam(param="glbar_leak", prediction=train_cell.passive_properties.g_bar_leak, bounds_variation=0.0, n_slices=1)
                    ],
                    filtered_out_features = ["no_spikes", "saturated"],
                    train_features=["i_trace_stats", "number_of_spikes", "spike_times", "spike_height_stats", "trough_times", "trough_height_stats"],
                    spike_threshold=0,
                    saturation_threshold=-55,
                    first_n_spikes=20,
                    prediction_eval_method='features',
                    random_state=random_seed,
                    save_file=f"{module_final_folder}/results/saved_metrics.json"
                )
            )
        )

        final_predicted_g_data_file = final_mod.run()

        final_mod.pickle_rf(final_mod.rf_model,f"{module_final_folder}/trained_rf.pkl")
        print(train_cell.predicted_g)

        from act import ACTPlot
        ACTPlot.plot_v_comparison(
            final_predicted_g_data_file, 
            module_final_folder, 
            sim_par["CI_amps"],
            sim_par["h_dt"]
            )

        ACTPlot.plot_fi_comparison(
            module_final_folder, 
            sim_par["CI_amps"]
            )

        from act.Metrics import Metrics

        metrics = Metrics()

        mean, stdev = metrics.save_interspike_interval_comparison(
            module_final_folder,
            final_predicted_g_data_file,
            sim_par["CI_amps"], 
            sim_par["h_dt"],
            first_n_spikes=5,
            save_file=f"{module_final_folder}/results/saved_metrics.json"
        )

        # HighRange
        #actual_g={"gbar_na3": 0.065, "gkdrbar_kdr": 0.043,"gbar_nap": 0.00055,"gmbar_im": 0.0053, "glbar_leak": 4.5e-5},
        # CenterRange
        #actual_g={"gbar_na3": 0.05, "gkdrbar_kdr": 0.03,"gbar_nap": 0.0004,"gmbar_im": 0.0038, "glbar_leak": 3.6e-5},
        # LowRange
        #actual_g={"gbar_na3": 0.035, "gkdrbar_kdr": 0.017,"gbar_nap": 0.00025,"gmbar_im": 0.0023, "glbar_leak": 2.7e-5},
        if cell_props == "low":
            actual_g={"gbar_na3": 0.035, "gkdrbar_kdr": 0.017,"gbar_nap": 0.00025,"gmbar_im": 0.0023, "glbar_leak": 2.7e-5}
        elif cell_props == "center":
            actual_g={"gbar_na3": 0.05, "gkdrbar_kdr": 0.03,"gbar_nap": 0.0004,"gmbar_im": 0.0038, "glbar_leak": 3.6e-5}
        elif cell_props == "high":
            actual_g={"gbar_na3": 0.065, "gkdrbar_kdr": 0.043,"gbar_nap": 0.00055,"gmbar_im": 0.0053, "glbar_leak": 4.5e-5} 
            
        metrics.save_prediction_g_mae(
            actual_g=actual_g,
            save_file=f"{module_final_folder}/results/saved_metrics.json"
        )

        metrics.save_feature_mae(
            module_final_folder,
            final_predicted_g_data_file,
            ["i_trace_stats", "number_of_spikes", "spike_times", "spike_height_stats", "trough_times", "trough_height_stats"],
            sim_par["h_dt"],
            first_n_spikes=5,
            save_file=f"{module_final_folder}/results/saved_metrics.json"
        )


        from act import ACTPlot as actplt

        g_names = ["gbar_na3", "gkdrbar_kdr","gbar_nap","gmbar_im"]

        for i in range(len(g_names)-1):
            actplt.plot_training_feature_mae_contour_plot(
                module_final_folder,
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
                results_filename=f"{module_final_folder}/results/Feature_MAE_Contour_Plot_{g_names[0]}_{g_names[i+1]}.png"
            )
            
        from act import ACTPlot as actplt

        g_names = ["gbar_na3", "gkdrbar_kdr","gbar_nap","gmbar_im"]

        for i in range(len(g_names)-1):
            actplt.plot_training_fi_mae_contour_plot(
                module_final_folder,
                sim_par["CI_amps"], 
                sim_par["CI_dur"],
                sim_par["CI_delay"],
                sim_par["h_dt"],
                index1=0,
                index2=i+1,
                g_names=g_names,
                results_filename=f"{module_final_folder}/results/FI_MAE_Contour_Plot_{g_names[0]}_{g_names[i+1]}.png"
            )
            
        from act import ACTPlot as actplt

        g_names = ["gbar_na3", "gkdrbar_kdr","gbar_nap","gmbar_im"]

        for i in range(len(g_names)-1):
            actplt.plot_training_v_mae_contour_plot(
                module_final_folder,
                sim_par["CI_amps"], 
                sim_par["CI_delay"],
                sim_par["h_dt"],
                index1=0,
                index2=i+1,
                g_names=g_names,
                results_filename=f"{module_final_folder}/results/Voltage_MAE_Contour_Plot_{g_names[0]}_{g_names[i+1]}.png"
            )

if __name__ == "__main__":
    run()