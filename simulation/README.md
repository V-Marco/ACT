# Running a simulation


All together

```
# 1. Generate target traces
python generate_target_traces.py

# 2. Build the network to generate our training parameters
python generate_traces.py build

# 3. Run the network (parallel)
# sbatch batch_generate_traces.sh
# OR
python generate_traces.py

# 4. Generate ARIMA summary statistics
NOCUDA=1 python generate_arma_stats.py

# 5. Run the simulation
python run_simulation.py

# 6. View runs statistics
python analyze_res.py


```
