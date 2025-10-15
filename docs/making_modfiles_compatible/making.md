# Making modfiles compatible with ACT

ACT supports activation functions specified in the Boltzmann ($V_{1/2}$, $k$) format and provides tools for conversion from other specifications. The following examples demonstrate how to convert modfiles from the alpha-beta specification to the Boltzmann specification.

## General algorithm

1. Transfer the activation function from the modfile and express it as a `Python` function.
2. Define a suitable voltage range. Use the `fit_boltzmann()` function from `act.modfiles` to fit a sigmoid and obtain $V_{1/2}$ and $k$.
3. Replace the activation function in the modfile with
```c
variable = 1.0 / (1.0 + exp(-(v - vhalf)/k))
```