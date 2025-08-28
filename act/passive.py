import numpy as np
from act.types import SettablePassiveProperties, GettablePassiveProperties

class ACTPassiveModule:
    """Estimate passive properties. 
    """

    @staticmethod
    def compute_spp(R_in: float, soma_area: float, tau: float, V_rest: float) -> SettablePassiveProperties:
        """
        Compute the reversal potential and maximum conductance of the leak channel and membrane capacitance.

        Parameters
        ----------
        R_in: float
            Target input resistance (Ohm).
        
        soma_area: float
            Target soma area or total cell area (cm2).
        
        tau: float
            Target membrane time constant (s).

        V_rest: float
            Target resting potential (mV).
        
        Returns
        -------
        spp: SettablePassiveProperties
            Computed settable passive properties.
        """
        spp = SettablePassiveProperties()
        spp.e_rev_leak = V_rest
        spp.g_bar_leak = ACTPassiveModule.compute_g_bar_leak(R_in, soma_area)
        spp.Cm = ACTPassiveModule.compute_Cm(spp.g_bar_leak, tau)
        return spp


    @staticmethod
    def compute_g_bar_leak(R_in: float, soma_area: float) -> float:
        """
        Compute maximum conductance of the leak channel.

        Parameters
        ----------
        R_in: float
            Input resistance (Ohm).
        
        soma_area: float
            Soma (or total) area (cm2).
        
        Returns
        -------
        g_bar_leak: float
            (S / cm2)
        """
        return (1 / R_in) / soma_area


    @staticmethod
    def compute_Cm(g_bar_leak: float, tau: float) -> float:
        """
        Compute membrane capacitance.

        Parameters
        ----------
        g_bar_leak: float
            Maximum conductance of the leak channel (S / cm2).
        
        tau: float
            Membrane time constant (s).
        
        Returns
        -------
        Cm: float
            Membrane capacitance (uF / cm2).
        
        """
        return tau * g_bar_leak * 1e6
    
    @staticmethod
    def compute_gpp(passive_V: np.ndarray, dt: float, I_t_start: float, I_t_end: float, I_amp: float) -> GettablePassiveProperties:
        """
        Estimate input resistance, lower and upper bounds of the membrane time constant, membrane resting potential and the sag ratio
        from a passive trace.

        Parameters
        ----------
        passive_V: np.ndarray
            Voltage trace under negative current injection
            
        dt: float
            Timestep (ms)
            
        I_t_start: float
            Current injection start time (ms).
        
        I_t_end: float
            Current injection end time (ms)
            
        I_amp: float
            Current injection amplitude (mV)
            
        Returns
        -------
        gpp: GettablePassiveProperties
            Class containing gettable passive properties.
        """

        index_V_rest = int(I_t_start / dt) - 1

        # If there is no h channel, V_final == V_trough
        index_V_trough = index_V_rest + np.argmin(passive_V[index_V_rest:])
        index_V_final = int(I_t_end / dt) - 1

        V_rest = passive_V[index_V_rest]
        V_trough = passive_V[index_V_trough]
        V_final = passive_V[index_V_final]

        # R_in
        R_in_rest_to_trough = (V_rest - V_trough) / (0 - I_amp)
        R_in_trough_to_final = (V_final - V_trough) / (0 - I_amp)
        R_in_rest_to_final = (V_rest - V_final) / (0 - I_amp)

        # Tau1
        V_tau1 = V_rest - (V_rest - V_trough) * 0.632
        index_V_tau1 = np.argmax(passive_V[index_V_rest:] < V_tau1)
        tau1 = index_V_tau1 * dt

        # Tau2
        V_tau2 = V_trough - (V_trough - V_final) * 0.632
        index_V_tau2 = np.argmax(passive_V[index_V_trough:] > V_tau2)
        tau2 = index_V_tau2 * dt

        # Tau3
        # Weighted average time constant
        w0 = (V_rest - V_trough)
        w1 = (V_final - V_trough)
        tau3 = (w0 * tau1 + w1 * tau2) / (w0 + w1)

        # Sag ratio
        sag = (V_final - V_trough) / (V_rest - V_trough)

        gpp = GettablePassiveProperties(
            R_in_rest_to_trough = R_in_rest_to_trough,
            R_in_trough_to_final = R_in_trough_to_final,
            R_in_rest_to_final = R_in_rest_to_final,
            tau_rest_to_trough = tau1,
            tau_trough_to_final = tau2,
            tau_avg = tau3,
            sag_ratio = sag,
            V_rest = V_rest
        )
        return gpp