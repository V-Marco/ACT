import numpy as np

'''
ACTSegregator
Responsible for taking a cell and ouptutting instructions for developing a segregated version.
'''

class ACTSegregator:

    def __init__(self):
        pass

    def _get_segregation_message_right(self, right_v: float, cutoff_v: float, slope: float, intercept: float) -> str:
        '''
        Generates instructions for modifying the .mod files with positive slope sigmoid activation functions.
        Parameters:
        -----------
        self
        
        right_v: float
            Right-most voltage involved in the linear segment cutoff for the activation function
            
        cutoff_v: float
            Voltage at which the activation function is forced to 0 mV.
        
        slope: float
            Slope of the linear segment connecting the voltage cutoff and the rest of the activation function
        
        intercept: float
            Intercept of the linear segment
                 
        Returns:
        -----------
        message: str
            Instructions for .mod file changes
        '''
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
        '''
        Generates instructions for modifying the .mod files with negative slope sigmoid activation functions.
        Parameters:
        -----------
        self
        
        left_v: float
            Left-most voltage involved in the linear segment cutoff for the activation function
            
        cutoff_v: float
            Voltage at which the activation function is forced to 0 mV.
        
        slope: float
            Slope of the linear segment connecting the voltage cutoff and the rest of the activation function
        
        intercept: float
            Intercept of the linear segment
                 
        Returns:
        -----------
        message: str
            Instructions for .mod file changes
        '''
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

    def segregate(self, v: np.ndarray, activation_curves: list, v_rest: float, dv_from_rest: float, extrapolate_dv: float = 2) -> list:
        '''
        Returns segregated activation functions in order of the input activation functions. Prints instructions on modifying .modfiles.
        Parameters:
        -----------
        self
        
        v: np.ndarray
            Voltage array (x values mV)
        
        activation_curves: list
            List of arrays where each array is the probability of activation for each activation curve.
            
        v_rest: float
            Resting membrane potential
        
        dv_from_rest: float
            Set deviation from V_rest for passive module offset
        
        extrapolate_dv: float
            Linear segment offset from cutoff to left or rightmost connector to the rest of the curve.
                 
        Returns:
        -----------
        segregated_activation_curves: list
            Segregated activation curves.
        '''
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
    
    # Methods for Cutoff Value Shifting 
    # DEPRECATED
    def _get_segregation_message_right_values(self, right_v: float, cutoff_v: float, slope: float, intercept: float):
        linear_right_start_v = np.round(right_v, 3)
        linear_right_slope = np.round(slope, 3)
        linear_right_intercept = np.round(intercept, 3)
        v_cutoff = np.round(cutoff_v, 3)
        return v_cutoff, linear_right_start_v, linear_right_slope, linear_right_intercept

    # DEPRECATED
    def _get_segregation_message_left_values(self, left_v: float, cutoff_v: float, slope: float, intercept: float):
        linear_left_start_v = np.round(left_v, 3)
        linear_left_slope = np.round(slope, 3)
        linear_left_intercept = np.round(intercept, 3)
        v_cutoff = np.round(cutoff_v, 3)
        
        return v_cutoff, linear_left_start_v, linear_left_slope, linear_left_intercept
    
    # DEPRECATED
    def segregate_with_cutoff_shift(self, v: np.ndarray, activation_curves: list, p_cutoff: float = 0.16, extrapolate_dv: float = 2, on_the_right = True, v_cutoff_shift: float = 0):

        if on_the_right == True:
            # Find the voltage cutoff value as min(v_cutoffs)
            cutoff_vs = []
            for curve in activation_curves:
                cutoff_vs.append(np.max(v[curve <= p_cutoff]))
            cutoff_v = np.min(cutoff_vs)
        else:
            # Find the voltage cutoff value as min(v_cutoffs)
            cutoff_vs = []
            for curve in activation_curves:
                cutoff_vs.append(np.min(v[curve <= p_cutoff]))
            cutoff_v = np.max(cutoff_vs)
            
        cutoff_v = cutoff_v + v_cutoff_shift

        # Segregate each a.c.
        segregated_ac_cutoff_values = []
        print("Providing the values for the linear component of the activation curve")

        for curve_id, curve in enumerate(activation_curves):

            if on_the_right == True:
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

                v_cutoff, linear_start_v, linear_slope, linear_intercept = self._get_segregation_message_right_values(right_v, cutoff_v, slope, intercept)

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

                v_cutoff, linear_start_v, linear_slope, linear_intercept = self._get_segregation_message_left_values(left_v, cutoff_v, slope, intercept)

            segregated_ac_cutoff_values.append([curve, v_cutoff, linear_start_v, linear_slope, linear_intercept])
        
        return segregated_ac_cutoff_values
