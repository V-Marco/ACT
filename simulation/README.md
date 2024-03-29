# Running a simulation


All together

```
# 0. Updated your selected_config in simulation_configs.py

# 1. Generate target traces
python generate_target_traces.py

# 2. Build the network to generate our training parameters
python generate_traces.py build

# 3. Run the network (parallel)
# sbatch batch_generate_traces.sh
# OR
python generate_traces.py

# 4. Generate ARIMA summary statistics (OPTIONAL - will be loaded by run_simulation if the ariama is generated)
python generate_arma_stats.py

# 5. Run the simulation
python run_simulation.py

# 6. View runs statistics (OLD)
python analyze_res.py
python plot_fi.py 

# can also plot additional fi curve to compare seg and orig
# python plot_fi.py --extra-trace-file ../othersimulation/output_Simple_Spiker_seg/original/traces.h5 --extra-trace-label 'Model ACT-Segregated'

```

One line - THIS WILL NEED TO BE RUN ONCE FOR *EACH* SEGREGATION MODULE. We loop through and learn parameters once at a time. Generating arima stats is optional, and usually only helpful for LTO/HTO modules.
```
python generate_target_traces.py && python generate_traces.py build && python generate_traces.py && python generate_arma_stats.py && python run_simulation.py
```
To generate final plots (after all segregation modules):
```
# lto and hto target files will be generated automatically if you specified use_lto_amps or use_hto_amps at any point during segregation
python generate_target_traces.py --ignore_segregation && python plot_learned_parameters.py
```
