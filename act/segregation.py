import numpy as np

'''
ACTSegregator
Responsible for taking a cell and ouptutting instructions for developing a segregated version.
'''

class ACTSegregator:

    def __init__(self):
        pass

    def _get_segregation_message_right(self, right_v: float, cutoff_v: float, slope: float, intercept: float):
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

    def _get_segregation_message_left(self, left_v: float, cutoff_v: float, slope: float, intercept: float):
        message = f"""
        :Segregation
        if (v > {np.round(left_v, 3)}) {{
        replace_with_var_name = {np.round(slope, 3)} * v + {np.round(intercept, 3)}
        }}
        if (v > {np.round(cutoff_v, 3)}) {{
        replace_with_var_name = 0
        }}
        """
        return message

    def segregate(self, v: np.ndarray, activation_curves: list, v_rest: float, dv_from_rest: float, extrapolate_dv: float = 2):

        # Find the voltage cutoff value
        cutoff_v = v_rest + dv_from_rest

        # Segregate each a.c.
        segregated_activation_curves = []
        print("Update the modfiles with the following:")

        for curve_id, curve in enumerate(activation_curves):

            print("----------")
            print(f"Activation curve {curve_id}:")

            if dv_from_rest > 0:
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

                print(self._get_segregation_message_right(right_v, cutoff_v, slope, intercept))

            else:
                # Find value to extrapolate to (guideline: -2-3 mV)
                left_v = cutoff_v - extrapolate_dv # (mV)
                left_p = np.max(curve[v >= left_v])
                
                # Compute the slope and intercept
                slope = (left_p - 0) / (left_v - cutoff_v)
                intercept = left_p - slope * left_v

                # Extrapolate
                curve[curve <= left_p] = v[curve <= left_p] * slope + intercept

                # Remove a.c. below the cutoff value
                curve[v >= cutoff_v] = 0

                print(self._get_segregation_message_left(left_v, cutoff_v, slope, intercept))

            segregated_activation_curves.append(curve)
        
        return segregated_activation_curves


