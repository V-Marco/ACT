import numpy as np
from scipy.optimize import minimize

from act.modfiles import SegmentableStateVariable

class ACTSegmenter:

    def __init__(self):
        self._v = np.linspace(-100, 40, 1000)
        self._linear_dv = 3

    def segment_v(self, modfile_group: list, v_threshold: float = -70):

        # Extract segmentable state variables and their modfiles
        segmentable = self._extract_segmentable(modfile_group) # [(modfile, svar)]

        for entry in segmentable:
            _, svar = entry

            # Set the threshold
            svar._segmented_v_threshold = v_threshold

            # Adjust the slope and vhalf
            svar = self._optimize_vhalf_k(svar, v_threshold)

            # Linearly extrapolate
            svar = self._linearly_extrapolate(svar, v_threshold)
        
        # Report
        self._report(segmentable)

    def segment_p(self, modfile_group: list, p_threshold: float = 0.01):

        # Extract segmentable state variables and their modfiles
        segmentable = self._extract_segmentable(modfile_group) # [(modfile, svar)]

        # Find the voltage cutoff value: this is the value at p of the rightmost
        # activation curve (minimizes overlap between modules)
        v_threshold = []
        for entry in segmentable:
            _, svar = entry
            p = svar.activation_function(self._v)

            if self._is_increasing(svar, svar.vhalf, svar.k):
                v_threshold.append(self._v[np.argmax(p >= p_threshold)])
            else:
                v_threshold.append(self._v[np.argmax(p <= p_threshold)])
        v_threshold = np.max(v_threshold)

        # Perform the rest of the segmentation
        self.segment_v(modfile_group, v_threshold)


    def _extract_segmentable(self, modfile_group: list):
        segmentable = []
        for modfile in modfile_group:
            for svar in modfile.state_variables:
                if isinstance(svar, SegmentableStateVariable):
                    segmentable.append((modfile, svar))
        return segmentable

    def _optimize_vhalf_k(self, svar: SegmentableStateVariable, v_threshold: float):

        # Find the vhalf and k that minimize activation probability at v_threshold
        def objective(params):
            vhalf, k = params
            return svar._activation_function(vhalf, k, v_threshold)
        
        result = minimize(
            objective, 
            x0 = [svar.vhalf, svar.k],
            method = 'L-BFGS-B',
            bounds = [svar.segmentation_vhalf_bounds, svar.segmentation_k_bounds]
        ).x

        # Adjust svar
        svar._segmented_vhalf = result[0]
        svar._segmented_k = result[1]
        
        return svar

    def _is_increasing(self, svar, vhalf, k):
        x1 = svar._activation_function(vhalf, k, 0 + self._linear_dv)
        x0 = svar._activation_function(vhalf, k, 0)
        return (x1 - x0) > 0

    def _linearly_extrapolate(self, svar: SegmentableStateVariable, v_threshold: float):
            
        if self._is_increasing(svar, svar._segmented_vhalf, svar._segmented_k):
            # Find value to extrapolate to (guideline: +3 mV)
            right_v = v_threshold + self._linear_dv # (mV)
            right_p = np.max(svar._activation_function(svar._segmented_vhalf, svar._segmented_k, self._v)[self._v <= right_v])
            
            # Compute the slope and intercept
            svar._segmented_linear_slope = (right_p - 0) / (right_v - v_threshold)
            svar._segmented_linear_intercept = right_p - svar._segmented_linear_slope * right_v
        else:
            # Find value to extrapolate to (guideline: -3 mV)
            left_v = v_threshold - self._linear_dv # (mV)
            left_p = np.max(svar._activation_function(svar._segmented_vhalf, svar._segmented_k, self._v)[self._v >= left_v])
            
            # Compute the slope and intercept
            svar._segmented_linear_slope = (left_p - 0) / (left_v - v_threshold)
            svar._segmented_linear_intercept = left_p - svar._segmented_linear_slope * left_v

        return svar

    def _report(self, segmented: list[tuple]):
        print("--------------------")
        print("Segmentation results")
        print("--------------------")
        print(f"Module voltage cutoff: {np.round(segmented[0][1]._segmented_v_threshold, 2)} mV")
        print("")
        print(f"Estimated parameters for activation functions:")
        
        for entry in segmented:
            modfile, svar = entry
            print(f"{modfile.channel_name}({svar.name}): vhalf = {svar._segmented_vhalf} mV, k = {svar._segmented_k} mV")
