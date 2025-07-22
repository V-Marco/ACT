import numpy as np

class ACTSegregator:
    """
    Segment the activation curves into functional modules.
    """

    def _get_segregation_message_right(self, right_v: float, cutoff_v: float, slope: float, intercept: float) -> str:
        """
        Generates instructions for modifying the .mod files with positive slope sigmoid activation functions.
        
        Parameters
        ----------
        right_v: float
            Right-most voltage involved in the linear segment cutoff for the activation function.
            
        cutoff_v: float
            Voltage at which the activation function is forced to 0 mV.
        
        slope: float
            Slope of the linear segment connecting the voltage cutoff and the rest of the activation function.
        
        intercept: float
            Intercept of the linear segment.
                 
        Returns
        -------
        message: str
            Instructions for .mod file changes
        """
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
        """
        Generates instructions for modifying the .mod files with negative slope sigmoid activation functions.
        
        Parameters
        ----------
        left_v: float
            Left-most voltage involved in the linear segment cutoff for the activation function.
            
        cutoff_v: float
            Voltage at which the activation function is forced to 0 mV.
        
        slope: float
            Slope of the linear segment connecting the voltage cutoff and the rest of the activation function.
        
        intercept: float
            Intercept of the linear segment.
                 
        Returns
        -------
        message: str
            Instructions for .mod file changes.
        """
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

    def segregate(self, v: np.ndarray, activation_curves: list, cutoff_v: float, extrapolate_dv: float = 2) -> list:
        """
        Prints instructions on modifying modfiles. Returns segregated activation functions. 
        
        Parameters
        ----------
        v: np.ndarray of shape (T,)
            The range of voltage values over which each activation curve is defined. 
        
        activation_curves: list[np.ndarray of shape (T,)]
            List of activation curves. Each activation curve is defined over v.

        cutoff_v: float
            Voltage value below/above which each activation function is set to 0.
        
        extrapolate_dv: float
            Voltage range to linearly extrapolate activation functions over.
                 
        Returns
        -------
        segregated_activation_curves: list
            Segregated activation curves.
        """
        # Segregate each a.c.
        segregated_activation_curves = []
        print("Update the modfiles with the following:")

        for curve_id, curve in enumerate(activation_curves):

            # Determine if the a.c. is increasing or decreasing (h-channel)
            is_increasing = curve[-1] - curve[0] > 0

            print("----------")
            print(f"Activation curve {curve_id}:")

            if is_increasing:
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
