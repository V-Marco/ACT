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

# 4. Generate ARIMA summary statistics
python generate_arma_stats.py

# 5. Run the simulation
python run_simulation.py

# 6. View runs statistics
python analyze_res.py
python plot_fi.py 

# can also plot additional fi curve to compare seg and orig
# python plot_fi.py --extra-trace-file ../othersimulation/output_Simple_Spiker_seg/original/traces.h5 --extra-trace-label 'Model ACT-Segregated'

```

One line
```
python generate_target_traces.py && python generate_traces.py build && python generate_traces.py && python generate_arma_stats.py && python run_simulation.py && python analyze_res.py && python plot_fi.py
```
