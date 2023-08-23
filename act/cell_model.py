from neuron import h


class CellModel:
    def __init__(self, hoc_file: str, cell_name: str):
        """
        Constructs a Cell object.

        Parameters:
        ----------
        hoc_file: str
            Name of the .hoc file that defines the cell.

        cell_name: str
            Name of the cell in the .hoc file.
        """

        # Load the .hoc file
        h.load_file(hoc_file)

        # Initialize the cell
        hoc_cell = getattr(h, cell_name)()
        self.all = hoc_cell.all
        self.soma = hoc_cell.soma

        # Initialize recorders
        self.t = h.Vector().record(h._ref_t)
        self.Vm = h.Vector().record(self.soma[0](0.5)._ref_v)

        # Init injection
        self.inj = h.IClamp(self.soma[0](0.5))
        self.I = h.Vector().record(h._ref_i)

    def set_parameters(self, parameter_list: list, parameter_values: list) -> None:
        for sec in self.all:
            for index, key in enumerate(parameter_list):
                setattr(sec, key, parameter_values[index])

    def apply_current_injection(self, amp: float, dur: float, delay: float) -> None:
        self.inj.amp = amp
        self.inj.dur = dur
        self.inj.delay = delay
