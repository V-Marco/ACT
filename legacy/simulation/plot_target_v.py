import json
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from act import utils
from simulation_configs import selected_config
import meta_sweep


if '--sweep' in sys.argv:
    selected_config = meta_sweep.get_meta_params_for_sweep()
output_dir = utils.get_output_folder_name(selected_config)

target_v_file = output_dir + "target/target_v.json"
with open(target_v_file,'r') as f:
    t = json.load(f)

for trace in t['traces']:
    plt.figure()
    plt.plot(trace)

plt.show()
