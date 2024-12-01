import numpy as np

class ACTSegregator:

    def __init__(self):
        pass

    def _get_segregation_message(self, right_v: float, cutoff_v: float, slope: float, intercept: float):
        message = f"""
        :Segregation
        if (v < {np.round(right_v, 3)}) {{
        replace_with_var_name = {np.round(slope, 3)} * v + {np.round(intercept, 3)}
        }}
        if (v < {np.round(cutoff_v, 3)}) {{
        replace_with_var_name = 0
        }}
        """
        return message

    def segregate(self, v: np.ndarray, activation_curves: list, p_cutoff: float = 0.16, extrapolate_dv: float = 2):

        # Find the voltage cutoff value as min(v_cutoffs)
        cutoff_vs = []
        for curve in activation_curves:
            cutoff_vs.append(np.max(v[curve <= p_cutoff]))
        cutoff_v = np.min(cutoff_vs)

        # Segregate each a.c.
        segregated_activation_curves = []
        print("Update the modfiles with the following:")
        for curve_id, curve in enumerate(activation_curves):

            # Find value to extrapolate to (guideline: +2-3 mV)
            right_v = cutoff_v + extrapolate_dv # (mV)
            right_p = np.max(curve[v <= right_v])
            
            # Compute the slope and intercept
            slope = (right_p - 0) / (right_v - cutoff_v)
            intercept = right_p - slope * right_v

            # Extrapolate
            curve[curve <= right_p] = v[curve <= right_p] * slope + intercept

            # Remove a.c. below the cutoff value
            curve[v <= cutoff_v] = 0

            segregated_activation_curves.append(curve)

            print("----------")
            print(f"Activation curve {curve_id}:")
            print(self._get_segregation_message(right_v, cutoff_v, slope, intercept))
        
        return segregated_activation_curves


