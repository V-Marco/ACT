from simulation_configs import selected_config
import sys
sys.path.append("../")
from act import simulator
import meta_sweep

if __name__ == "__main__":
    if '--sweep' in sys.argv:
        selected_config = meta_sweep.get_meta_params_for_sweep()
    print("TRAINING MODEL")
    p = simulator.run(selected_config, subprocess=False)
