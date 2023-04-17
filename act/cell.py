from neuron import h
import matplotlib.pyplot as plt
from scipy import signal

class Cell:

    def __init__(self, hoc_file: str, cell_name: str) -> None:
        '''
        Constructs a Cell object.

        Parameters:
        ----------
        hoc_file: str
            Name of the .hoc file that defines the cell.

        cell_name: str
            Name of the cell in the .hoc file.
        '''

        # Load the .hoc file
        h.load_file(hoc_file)

        # Get the cell by name
        invoke_cell = getattr(h, cell_name)
        self.cell = invoke_cell()

        # Initialize vectors to track membrane voltage
        self.mem_potential = h.Vector()
        self.time = h.Vector()

        # Record time and membrane potential.
        self.time.record(h._ref_t)
        self.mem_potential.record(self.get_recording_section()._ref_v)

    def get_recording_section(self):
        return self.cell.soma[0](0.5)
    
    def set_parameters(self, parameter_list: list, parameter_values: list) -> None:
        '''
        Utility method to set the cell's parameters in a name-value fashion.

        Parameters:
        ----------
        parameter_list: list
            List with parameters' names.

        parameter_values: list
            List with parameters' values.
        '''
        for sec in self.cell.all:
            for index, key in enumerate(parameter_list):
                setattr(sec, key, parameter_values[index])

    def resample(self):
        '''
        Resamples the membrane voltage and time vectors.
        '''
        return signal.resample(x = self.mem_potential.as_numpy(), num = 32**2, t = self.time.as_numpy())

    def plot_potential(self) -> None:
        '''
        Plot the membrane vs time graph based on whatever is in membrane voltage and time vectors.
        '''
        plt.close()
        plt.figure(figsize = (20, 5))
        plt.plot(self.time, self.mem_potential)
        plt.xlabel('Time')
        plt.ylabel('Membrane Potential')
        plt.show()
