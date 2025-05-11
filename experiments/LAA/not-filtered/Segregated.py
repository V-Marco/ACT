
def run():
    num_seeds = 1
    for seed in range(42,42+num_seeds): 
        # Set the path
        import sys
        sys.path.append("/home/ubuntu/ACT")

        from act.cell_model import ACTCellModel
        from act.simulator import ACTSimulator
        from act.act_types import SimulationParameters, ConstantCurrentInjection, FilterParameters, ConductanceOptions, OptimizationParameters
        from act.data_processing import ACTDataProcessor as dp
        from act.act_module import ACTModule, ACTModuleParameters

        import numpy as np
        import matplotlib.pyplot as plt
        
        random_seed = seed
        num_slices = 3
        experiment_folder = f"output/LAA_segregated-{num_slices}/{random_seed}"
        target_folder = experiment_folder + "/target"
        module1_folder = experiment_folder + "/module_1"
        module2_folder = experiment_folder + "/module_2"
        module_final_folder = experiment_folder + "/module_final"

        target_cell = ACTCellModel(
            path_to_hoc_file="/home/ubuntu/ACT/data/LAA/seg/template.hoc",
            path_to_mod_files="/home/ubuntu/ACT/data/LAA/seg/modfiles",
            cell_name="Cell_A_seg",
            passive=[],
            active_channels=["gbar_nap",
                            "gmbar_im", 
                            "gcabar_cadyn", 
                            "gsAHPbar_sAHP", 
                            "gkabar_kap",
                            "gbar_na3",
                            "gkdrbar_kdr"]
        )

        sim_par= SimulationParameters(
                    h_v_init=-65.0,  # Consistent with nrngui settings
                    h_tstop=1000,     # Simulation time in ms
                    h_dt=0.1,      # Time step in ms
                    h_celsius=6.3,   # Temperature in degrees Celsius
                    CI=[ConstantCurrentInjection(amp=0.040,dur=800,delay=100, lto_hto=1),
                        ConstantCurrentInjection(amp=0.045,dur=800,delay=100, lto_hto=1),
                        ConstantCurrentInjection(amp=0.050,dur=800,delay=100, lto_hto=1),
                        ConstantCurrentInjection(amp=0.055,dur=800,delay=100, lto_hto=1),
                        ConstantCurrentInjection(amp=0.060,dur=800,delay=100, lto_hto=1),
                        ConstantCurrentInjection(amp=0.0,dur=800,delay=100),
                        ConstantCurrentInjection(amp=0.1,dur=800,delay=100),
                        ConstantCurrentInjection(amp=0.2,dur=800,delay=100),
                        ConstantCurrentInjection(amp=0.3,dur=800,delay=100),
                        ConstantCurrentInjection(amp=0.4,dur=800,delay=100),
                        ConstantCurrentInjection(amp=4.5,dur=800,delay=100, lto_hto=1),
                        ConstantCurrentInjection(amp=5.0,dur=800,delay=100, lto_hto=1),
                        ConstantCurrentInjection(amp=5.5,dur=800,delay=100, lto_hto=1),
                        ConstantCurrentInjection(amp=6.0,dur=800,delay=100, lto_hto=1),
                        ConstantCurrentInjection(amp=6.5,dur=800,delay=100, lto_hto=1)],
                    set_g_to=[]
                )
        
        # Set simulations
        simulator = ACTSimulator(output_folder_name = ".")

        # LTO
        for sim_idx, amp_value in enumerate([0.04, 0.045, 0.05, 0.055, 0.06]):
            sim_params = SimulationParameters(
                sim_name = target_folder, 
                sim_idx = sim_idx, 
                h_v_init=-65,
                h_celsius = 6.3,
                h_dt = 0.1,
                h_tstop = 1000,
                CI = [ConstantCurrentInjection(amp = amp_value, dur = 800, delay = 100, lto_hto = 1)])

            simulator.submit_job(target_cell, sim_params)
            
        # Normal
        for sim_idx, amp_value in enumerate([0.0, 0.1, 0.2, 0.3, 0.4]):
            sim_params = SimulationParameters(
                sim_name = target_folder, 
                sim_idx = sim_idx+5, 
                h_v_init=-65,
                h_celsius = 6.3,
                h_dt = 0.1,
                h_tstop = 1000,
                CI = [ConstantCurrentInjection(amp = amp_value, dur = 800, delay = 100, lto_hto = 0)])
            
            simulator.submit_job(target_cell, sim_params)

        #HTO
        for sim_idx, amp_value in enumerate([4.5, 5.0, 5.5, 6.0, 6.5]):
            sim_params = SimulationParameters(
                sim_name = target_folder, 
                sim_idx = sim_idx+10, 
                h_v_init=-65,
                h_celsius = 6.3,
                h_dt = 0.1,
                h_tstop = 1000,
                CI = [ConstantCurrentInjection(amp = amp_value, dur = 800, delay = 100, lto_hto = 1)])

            simulator.submit_job(target_cell, sim_params)
            
        simulator.run_jobs(15)
        
        dp.combine_data(target_folder)
        
        train_cell = ACTCellModel(
            path_to_hoc_file="/home/ubuntu/ACT/data/LAA/seg/template.hoc",
            path_to_mod_files="/home/ubuntu/ACT/data/LAA/seg/modfiles",
            cell_name="Cell_A_seg",
            passive=[],
            active_channels=["gbar_nap",
                            "gmbar_im", 
                            "gcabar_cadyn", 
                            "gsAHPbar_sAHP", 
                            "gkabar_kap",
                            "gbar_na3",
                            "gkdrbar_kdr"]
            )
        
        random_state = np.random.RandomState(123)

        gmbar_im = 0.002
        gbar_nap= 0.000142
        gbar_na3=0.03
        gkdrbar_kdr=0.0015
        gcabar_cadyn = 6e-5
        gsAHPbar_sAHP = 0.009
        gkabar_kap = 0.000843
        
        gbar_nap_range = (gbar_nap - random_state.uniform(0, gbar_nap / 2), gbar_nap + random_state.uniform(0, gbar_nap / 2))
        gmbar_im_range = (gmbar_im - random_state.uniform(0, gmbar_im / 2), gmbar_im + random_state.uniform(0, gmbar_im / 2))
        gbar_na3_range = (gbar_na3 - random_state.uniform(0, gbar_na3 / 2), gbar_na3 + random_state.uniform(0, gbar_na3 / 2))
        gkdrbar_kdr_range = (gkdrbar_kdr - random_state.uniform(0, gkdrbar_kdr / 2), gkdrbar_kdr + random_state.uniform(0, gkdrbar_kdr / 2))
        gcabar_cadyn_range = (gcabar_cadyn - random_state.uniform(0, gcabar_cadyn / 2), gcabar_cadyn + random_state.uniform(0, gcabar_cadyn / 2))
        gsAHPbar_sAHP_range = (gsAHPbar_sAHP - random_state.uniform(0, gsAHPbar_sAHP / 2), gsAHPbar_sAHP + random_state.uniform(0, gsAHPbar_sAHP / 2))
        gkabar_kap_range = (gkabar_kap - random_state.uniform(0, gkabar_kap / 2), gkabar_kap + random_state.uniform(0, gkabar_kap / 2))

        # MODULE 1 (BURSTING)
        mod1 = ACTModule(
            ACTModuleParameters(
                module_folder_name=module1_folder,
                cell= train_cell,
                target_traces_file = f"{target_folder}/combined_out.npy",
                sim_params= sim_par,
                optim_params= OptimizationParameters(
                    conductance_options = [
                        ConductanceOptions(variable_name="gbar_nap", low=gbar_nap_range[0], high= gbar_nap_range[1], n_slices=num_slices),
                        ConductanceOptions(variable_name="gmbar_im", low=gmbar_im_range[0], high= gmbar_im_range[1],  n_slices=num_slices),
                        ConductanceOptions(variable_name="gcabar_cadyn", low= gcabar_cadyn_range[0], high= gcabar_cadyn_range[1], n_slices=num_slices),
                        ConductanceOptions(variable_name="gsAHPbar_sAHP", low= gsAHPbar_sAHP_range[0], high= gsAHPbar_sAHP_range[1], n_slices=num_slices),
                        ConductanceOptions(variable_name="gkabar_kap", low= gkabar_kap_range[0], high= gkabar_kap_range[1], n_slices=num_slices),
                        ConductanceOptions(variable_name="gbar_na3", low=gbar_na3_range[0], high= gbar_na3_range[1], n_slices=1),
                        ConductanceOptions(variable_name="gbar_kdr", low=gkdrbar_kdr_range[0], high= gkdrbar_kdr_range[1], n_slices=1)
                    ],
                    train_features=["i_trace_stats", "number_of_spikes", "spike_times", "spike_height_stats", "trough_times", "trough_height_stats", "lto-hto_amplitude", "lto-hto_frequency"],
                    prediction_eval_method='features',
                    save_file=f"{module1_folder}/results/saved_metrics.json",
                    spike_threshold=-50
                )
            )
        )
        
        mod1.run()
        
        bounds_variation = 0.15
        
        # MODULE 2 (SPIKING)
        mod2 = ACTModule(
            ACTModuleParameters(
                module_folder_name=module2_folder,
                cell= train_cell,
                target_traces_file = f"{target_folder}/combined_out.npy",
                sim_params= sim_par,
                optim_params= OptimizationParameters(
                    conductance_options = [
                        ConductanceOptions(variable_name="gbar_nap", prediction=train_cell.prediction["gbar_nap"], bounds_variation=train_cell.prediction["gbar_nap"]*bounds_variation, num_slices=num_slices),
                        ConductanceOptions(variable_name="gmbar_im", prediction=train_cell.prediction["gmbar_im"], bounds_variation=train_cell.prediction["gmbar_im"]*bounds_variation, num_slices=num_slices),
                        ConductanceOptions(variable_name="gcabar_cadyn", prediction=train_cell.prediction["gcabar_cadyn"], bounds_variation=train_cell.prediction["gcabar_cadyn"]*bounds_variation, num_slices=num_slices),
                        ConductanceOptions(variable_name="gsAHPbar_sAHP", prediction=train_cell.prediction["gsAHPbar_sAHP"], bounds_variation=train_cell.prediction["gsAHPbar_sAHP"]*bounds_variation, num_slices=num_slices),
                        ConductanceOptions(variable_name="gkabar_kap", prediction=train_cell.prediction["gkabar_kap"], bounds_variation=train_cell.prediction["gkabar_kap"]*bounds_variation, num_slices=num_slices),
                        ConductanceOptions(variable_name="gbar_na3", low=gbar_na3_range[0], high= gbar_na3_range[1], n_slices=num_slices),
                        ConductanceOptions(variable_name="gbar_kdr", low=gkdrbar_kdr_range[0], high= gkdrbar_kdr_range[1], n_slices=num_slices)
                    ],
                    train_features=["i_trace_stats", "number_of_spikes", "spike_times", "spike_height_stats", "trough_times", "trough_height_stats", "lto-hto_amplitude", "lto-hto_frequency"],
                    prediction_eval_method='features',
                    save_file=f"{module2_folder}/results/saved_metrics.json",
                    spike_threshold=-50
                )
            )
        )
        
        mod2.run()
        
        # FINAL MODULE (Fine Tuning)
        bounds_variation = 0.15
        mod_fin = ACTModule(
            ACTModuleParameters(
                module_folder_name=module_final_folder,
                cell= train_cell,
                target_traces_file = f"{target_folder}/combined_out.npy",
                sim_params= sim_par,
                optim_params= OptimizationParameters(
                    conductance_options = [
                        ConductanceOptions(variable_name="gbar_nap", prediction=train_cell.prediction["gbar_nap"], bounds_variation=train_cell.prediction["gbar_nap"]*bounds_variation, num_slices=num_slices),
                        ConductanceOptions(variable_name="gmbar_im", prediction=train_cell.prediction["gmbar_im"], bounds_variation=train_cell.prediction["gmbar_im"]*bounds_variation, num_slices=num_slices),
                        ConductanceOptions(variable_name="gcabar_cadyn", prediction=train_cell.prediction["gcabar_cadyn"], bounds_variation=train_cell.prediction["gcabar_cadyn"]*bounds_variation, num_slices=num_slices),
                        ConductanceOptions(variable_name="gsAHPbar_sAHP", prediction=train_cell.prediction["gsAHPbar_sAHP"], bounds_variation=train_cell.prediction["gsAHPbar_sAHP"]*bounds_variation, num_slices=num_slices),
                        ConductanceOptions(variable_name="gkabar_kap", prediction=train_cell.prediction["gkabar_kap"], bounds_variation=train_cell.prediction["gkabar_kap"]*bounds_variation, num_slices=num_slices),
                        ConductanceOptions(variable_name="gbar_na3", prediction=train_cell.prediction["gbar_na3"], bounds_variation=train_cell.prediction["gbar_na3"]*bounds_variation, num_slices=num_slices),
                        ConductanceOptions(variable_name="gbar_kdr", prediction=train_cell.prediction["gbar_kdr"], bounds_variation=train_cell.prediction["gbar_kdr"]*bounds_variation, num_slices=num_slices)
                    ],
                    train_features=["i_trace_stats", "number_of_spikes", "spike_times", "spike_height_stats", "trough_times", "trough_height_stats", "lto-hto_amplitude", "lto-hto_frequency"],
                    prediction_eval_method='features',
                    save_file=f"{module_final_folder}/results/saved_metrics.json",
                    spike_threshold=-50
                )
            )
        )
        
        predicted_g_data_file = mod_fin.run()
        
        
        from act import act_plot
        act_plot.plot_v_comparison(
            module_final_folder,
            predicted_g_data_file, 
            sim_par.CI,
            sim_par.h_dt
            )

        act_plot.plot_fi_comparison(
            module_final_folder, 
            sim_par.CI
            )
        
        from act.metrics import Metrics

        metrics = Metrics()

        mean, stdev = metrics.save_interspike_interval_comparison(
            module_final_folder,
            predicted_g_data_file,
            sim_par.CI, 
            sim_par.h_dt,
            first_n_spikes=5,
            save_file=f"{module_final_folder}/results/saved_metrics.json"
        )

        '''
        gmbar_im = 0.002
        gbar_nap= 0.000142
        gbar_na3=0.03
        gkdrbar_kdr=0.0015
        gcabar_cadyn = 6e-5
        gsAHPbar_sAHP = 0.009
        gkabar_kap = 0.000843
        ghdbar_hd=2.3e-05
        glbar_leak = 5.5e-5
        '''

        actual_g={"gbar_nap": 0.000142, "gmbar_im": 0.002, "gcabar_cadyn": 6e-5, "gsAHPbar_sAHP": 0.009, "gkabar_kap": 0.000843, "gbar_na3":0.03, "gkdrbar_kdr":0.0015}

            
        metrics.save_prediction_g_mae(
            actual_g=actual_g,
            save_file=f"{module_final_folder}/results/saved_metrics.json"
        )

        metrics.save_feature_mae(
            module_final_folder,
            predicted_g_data_file,
            ["i_trace_stats", "number_of_spikes", "spike_times", "spike_height_stats", "trough_times", "trough_height_stats", "lto-hto_amplitude", "lto-hto_frequency"],
            sim_par.h_dt,
            first_n_spikes=5,
            CI_settings=sim_par.CI,
            save_file=f"{module_final_folder}/results/saved_metrics.json"
        )


        from act import act_plot as actplt

        g_names = ["gbar_nap",
                    "gmbar_im", 
                    "gcabar_cadyn", 
                    "gsAHPbar_sAHP", 
                    "gkabar_kap",
                    "gbar_na3",
                    "gkdrbar_kdr"]

        for i in range(len(g_names)-1):
            actplt.plot_training_feature_mae_contour_plot(
                module_final_folder,
                sim_par.CI,
                sim_par.CI[0].delay,
                sim_par.h_dt,
                index1=0,
                index2=i+1,
                g_names=g_names,
                train_features=["i_trace_stats", "number_of_spikes", "spike_times", "spike_height_stats", "trough_times", "trough_height_stats", "lto-hto_amplitude", "lto-hto_frequency"],
                threshold=0,
                first_n_spikes=20,
                num_levels=100,
                results_filename=f"{module_final_folder}/results/Feature_MAE_Contour_Plot_{g_names[0]}_{g_names[i+1]}.png"
            )
            
        from act import act_plot as actplt

        g_names = ["gbar_nap",
                    "gmbar_im", 
                    "gcabar_cadyn", 
                    "gsAHPbar_sAHP", 
                    "gkabar_kap",
                    "gbar_na3",
                    "gkdrbar_kdr"]

        for i in range(len(g_names)-1):
            actplt.plot_training_fi_mae_contour_plot(
                module_final_folder,
                sim_par.CI, 
                sim_par.CI[0].delay,
                sim_par.h_dt,
                index1=0,
                index2=i+1,
                g_names=g_names,
                spike_threshold=0,
                results_filename=f"{module_final_folder}/results/FI_MAE_Contour_Plot_{g_names[0]}_{g_names[i+1]}.png"
            )
            
        from act import act_plot as actplt

        g_names = ["gbar_nap",
                    "gmbar_im", 
                    "gcabar_cadyn", 
                    "gsAHPbar_sAHP", 
                    "gkabar_kap",
                    "gbar_na3",
                    "gkdrbar_kdr"]

        for i in range(len(g_names)-1):
            actplt.plot_training_v_mae_contour_plot(
                module_final_folder,
                sim_par.CI, 
                sim_par.CI[0].delay,
                sim_par.h_dt,
                index1=0,
                index2=i+1,
                g_names=g_names,
                results_filename=f"{module_final_folder}/results/Voltage_MAE_Contour_Plot_{g_names[0]}_{g_names[i+1]}.png"
            )
    
    
if __name__ == "__main__":
    run()