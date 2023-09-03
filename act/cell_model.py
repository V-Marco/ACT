from neuron import h

from act import act_types


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

        # Passive properties
        self.gleak_var = None
        self.g_bar_leak = None

    def set_parameters(self, parameter_list: list, parameter_values: list) -> None:
        for sec in self.all:
            for index, key in enumerate(parameter_list):
                setattr(sec, key, parameter_values[index])

    def apply_current_injection(self, amp: float, dur: float, delay: float) -> None:
        self.inj.amp = amp
        self.inj.dur = dur
        self.inj.delay = delay

    def set_passive_properties(
        self, passive_properties: act_types.PassiveProperties
    ) -> None:
        if passive_properties:
            v_rest = passive_properties.get("v_rest")  # mV
            r_in = passive_properties.get("r_in")  # Ohms
            tau = passive_properties.get("tau")  # ms

            gleak_var = passive_properties.get("leak_conductance_variable")
            eleak_var = passive_properties.get("leak_reversal_variable")

            # Assuming Vrest is within the range for ELeak
            # ELeak = Vrest
            if v_rest and eleak_var:
                print(f"Setting {eleak_var} = {v_rest}")
                self.set_parameters([eleak_var], [v_rest])
            else:
                print(
                    f"Skipping analytical setting of e_leak variable. Cell v_rest and/or leak_reversal_variable not specified in config."
                )

            for sec in self.all:
                # Rin = 1/(Area*g_bar_leak)
                # g_bar leak = 1/(Rin*Area)
                area = sec(
                    0.5
                ).area()  # have to specify the location to get access to area
                if r_in:
                    g_bar_leak = 1 / (r_in * area) * 1e2  # area m to cm?
                    print(f"Setting {sec}.{gleak_var} = {g_bar_leak:.8f}")
                    setattr(sec, gleak_var, g_bar_leak)
                    if sec == self.soma:
                        g_bar_leak = g_bar_leak

                    if tau:
                        # tau = (R*C) = R*Area*cm
                        # cm = tau*g_bar_leak*Area
                        # cm/cm2 = (tau*g_bar_leak*Area)/Area = tau*g_bar_leak
                        cm = tau * g_bar_leak * 1e3  # tau ms->s
                        print(f"Setting {sec}.cm = {cm:.8f}")
                        setattr(sec, "cm", cm)
                    else:
                        print(
                            f"Skipping analytical setting of cm variable. Cell tau not specified in config."
                        )
                else:
                    print(
                        f"Skipping analytical setting of gleak and cm variables. Cell r_in not specified in config."
                    )
        else:
            print(
                f"Skipping analytical setting of passive properties, no cell passive_properties specified in config."
            )
