import json
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from act import utils
from simulation_configs import selected_config
import os

output_dir = utils.get_output_folder_name(selected_config)

if(selected_config["run_mode"] == "segregated"):
    segregation_index = utils.get_segregation_index(selected_config)
    segregation_dir = f"seg_module_{segregation_index+1}/"
    model_data_dir = os.path.join(output_dir, segregation_dir)
else:
    model_data_dir = output_dir
with open(model_data_dir + "train_stats_repeat_1.json", "r") as fp:
    d = json.load(fp)

plt.plot(d["train_loss"], label="Train Loss")
plt.plot(d["test_loss"], label="Test Loss")
plt.legend()
plt.show()
