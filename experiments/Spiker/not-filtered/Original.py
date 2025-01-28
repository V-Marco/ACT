
def run():
    num_seeds = 5
    for seed in range(42,42+num_seeds): 
        from act.SyntheticGenerator import SyntheticGenerator
        from act.act_types import SimulationParameters, OptimizationParameters, ConstantCurrentInjection
        from act.cell_model import TargetCell
        from act.module_parameters import ModuleParameters

        random_seed = seed
        num_slices = 10

        experiment_folder = f"output/Spiker_Original_sl-{num_slices}/{random_seed}"
        target_folder = experiment_folder + "/target"
        final_folder = experiment_folder + "/final"

        target_cell = TargetCell(
            path_to_hoc_file="/home/mwsrgf/proj/ACT/data/Spiker/orig/template.hoc",
            path_to_mod_files="/home/mwsrgf/proj/ACT/data/Spiker/orig/modfiles",
            cell_name="Spiker",
            active_channels = ["gnabar_hh_spiker", "gkbar_hh_spiker", "gl_hh_spiker"]
        )

        sim_par= SimulationParameters(
                    h_v_init=-65.0,  # Consistent with nrngui settings
                    h_tstop=500,     # Simulation time in ms
                    h_dt=0.1,        # Time step in ms
                    h_celsius=6.3,   # Temperature in degrees Celsius
                    CI = [ConstantCurrentInjection(amp=0.1,dur=300,delay=100, lto_hto=0),
                        ConstantCurrentInjection(amp=0.2,dur=300,delay=100, lto_hto=0),
                        ConstantCurrentInjection(amp=0.3,dur=300,delay=100, lto_hto=0)],
                    set_g_to=[]
                )

        sg = SyntheticGenerator(
            ModuleParameters(
                module_folder_name=target_folder,
                cell= target_cell,
                sim_params= sim_par
            )
        )

        sg.generate_synthetic_target_data("target_data.csv")

        import random

        random.seed(random_seed)

        glbar_leak = 0.0003
        gbar_na=0.12
        gkdrbar_kdr=0.036

        glbar_low_offset = random.uniform(0,glbar_leak/2)
        na_low_offset = random.uniform(0,gbar_na/2)
        kdr_low_offset = random.uniform(0,gkdrbar_kdr/2)

        glbar_low = glbar_leak - glbar_low_offset
        glbar_high = glbar_leak + ((glbar_leak/2) - glbar_low_offset)

        na_low = gbar_na - na_low_offset
        na_high = gbar_na + ((gbar_na/2) - na_low_offset)

        kdr_low = gkdrbar_kdr - kdr_low_offset
        kdr_high = gkdrbar_kdr + ((gkdrbar_kdr/2) - kdr_low_offset)



        print(f"glbar: ({glbar_low},{glbar_high}) -- TRUE: {glbar_leak}")
        print(f"na: ({na_low},{na_high}) -- TRUE: {gbar_na}")
        print(f"kdr: ({kdr_low},{kdr_high}) -- TRUE: {gkdrbar_kdr}")


        print("-------------")
        print("Range (high - low) ")
        print(f"leak range: {glbar_high - glbar_low} -- 50% TRUE: {glbar_leak/2}")
        print(f"na range: {na_high - na_low} -- 50% TRUE: {gbar_na/2}")
        print(f"kdr range: {kdr_high - kdr_low} -- 50% TRUE: {gkdrbar_kdr/2}")
        
        from act.ACTModule import ACTModule
        from act.act_types import OptimizationParameters, ConductanceOptions, FilterParameters
        from act.cell_model import TrainCell

        train_cell = TrainCell(
            path_to_hoc_file="/home/mwsrgf/proj/ACT/data/Spiker/orig/template.hoc",
            path_to_mod_files="/home/mwsrgf/proj/ACT/data/Spiker/orig/modfiles",
            cell_name="Spiker",
            active_channels = ["gnabar_hh_spiker", "gkbar_hh_spiker", "gl_hh_spiker"]
        )

        mod = ACTModule(
            ModuleParameters(
                module_folder_name=final_folder,
                cell= train_cell,
                target_traces_file = f"{target_folder}/target_data.csv",
                sim_params= sim_par,
                optim_params= OptimizationParameters(
                    conductance_options= [
                        ConductanceOptions(variable_name="gnabar_hh_spiker", low=na_low, high=na_high, n_slices=num_slices),
                        ConductanceOptions(variable_name="gkbar_hh_spiker", low=kdr_low, high=kdr_high, n_slices=num_slices),
                        ConductanceOptions(variable_name="gl_hh_spiker", low=glbar_low, high=glbar_high, n_slices=num_slices),
                        
                    ],
                    train_features=["i_trace_stats", "number_of_spikes", "spike_times", "spike_height_stats", "number_of_troughs", "trough_times", "trough_height_stats"],
                    prediction_eval_method='fi_curve',
                    spike_threshold=0,
                    filter_parameters=FilterParameters(
                        saturation_threshold=-55,
                        window_of_inspection=(120,400)
                    ),
                    first_n_spikes=20,
                    random_state=random_seed,
                    save_file=f"{final_folder}/results/saved_metrics.json"
                )
            )
        )
        
        predicted_g_data_file = mod.run()
        
        mod.pickle_rf(mod.rf_model,f"{final_folder}/trained_rf.pkl")
        
        from act import ACTPlot
        ACTPlot.plot_v_comparison(
            final_folder,
            predicted_g_data_file, 
            sim_par.CI,
            sim_par.h_dt
            )

        ACTPlot.plot_fi_comparison(
            final_folder, 
            sim_par.CI
            )
        
        from act.Metrics import Metrics

        metrics = Metrics()

        mean, stdev = metrics.save_interspike_interval_comparison(
            final_folder,
            predicted_g_data_file,
            sim_par.CI, 
            sim_par.h_dt,
            first_n_spikes=5,
            save_file=f"{final_folder}/results/saved_metrics.json"
        )

        '''
        "gnabar_hh_orig" = 0.12
        "gkbar_hh_orig"=0.036
        "gl_hh_orig"=0.0003
        '''

        actual_g={"gnabar_hh_spiker": 0.12,"gkbar_hh_spiker": 0.036, "gl_hh_spiker": 0.0003}

            
        metrics.save_prediction_g_mae(
            actual_g=actual_g,
            save_file=f"{final_folder}/results/saved_metrics.json"
        )

        metrics.save_feature_mae(
            final_folder,
            predicted_g_data_file,
            ["i_trace_stats", "number_of_spikes", "spike_times", "spike_height_stats", "trough_times", "trough_height_stats", "lto-hto_amplitude", "lto-hto_frequency"],
            sim_par.h_dt,
            first_n_spikes=5,
            CI_settings=sim_par.CI,
            save_file=f"{final_folder}/results/saved_metrics.json"
        )

        from act import ACTPlot as actplt

        g_names = ["gnabar_hh_spiker", "gkbar_hh_spiker", "gl_hh_spiker"]

        for i in range(len(g_names)-1):
            actplt.plot_training_feature_mae_contour_plot(
                final_folder,
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
                results_filename=f"{final_folder}/results/Feature_MAE_Contour_Plot_{g_names[0]}_{g_names[i+1]}.png"
            )
            
        from act import ACTPlot as actplt

        g_names = ["gnabar_hh_spiker", "gkbar_hh_spiker", "gl_hh_spiker"]

        for i in range(len(g_names)-1):
            actplt.plot_training_fi_mae_contour_plot(
                final_folder,
                sim_par.CI, 
                sim_par.CI[0].dur,
                sim_par.CI[0].delay,
                sim_par.h_dt,
                index1=0,
                index2=i+1,
                g_names=g_names,
                results_filename=f"{final_folder}/results/FI_MAE_Contour_Plot_{g_names[0]}_{g_names[i+1]}.png"
            )

        from act import ACTPlot as actplt

        g_names = ["gnabar_hh_spiker", "gkbar_hh_spiker", "gl_hh_spiker"]


        for i in range(len(g_names)-1):
            actplt.plot_training_v_mae_contour_plot(
                final_folder,
                sim_par.CI, 
                sim_par.CI[0].delay,
                sim_par.h_dt,
                index1=0,
                index2=i+1,
                g_names=g_names,
                results_filename=f"{final_folder}/results/Voltage_MAE_Contour_Plot_{g_names[0]}_{g_names[i+1]}.png"
            )


        
if __name__ == "__main__":
        run()