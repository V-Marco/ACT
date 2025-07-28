# Workflow

Here we provide a step-by-step example of using the ACT workflow to tune a detailed Layer-5 cell model.

## Inputs

We assume that the following target biological data were provided (recorded from a [mouse Layer-5 primary visual area cell](https://celltypes.brain-map.org/experiment/electrophysiology/599393756)):
- The cell has active ion channels in the dendrites.
- Rheobase: 190 pA
- Input resistance: 167 MOhm
- Membrane time constant: 26.6 ms
- Resting potential: -71.7 mV
- FI curve:
  
| I (pA) | F (Hz) |
| -----: | -----: |
|  150   | 0      |
|  190   | 7      |
|  230   | 12     |
|  270   | 15     |
|  330   | 21     |