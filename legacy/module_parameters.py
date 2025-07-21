from act.act_types import SimulationParameters, OptimizationParameters
from act.cell_model import ACTCellModel
from dataclasses import dataclass

# This dataclass is necessary for inputs to the ACTModule

@dataclass
class ModuleParameters:
    module_folder_name: str = None
    target_traces_file: str = None
    cell: ACTCellModel = None
    sim_params: SimulationParameters = None
    optim_params: OptimizationParameters = None