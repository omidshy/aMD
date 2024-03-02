
# Python codes for MD simulation analysis

A set of programs to compute physical properties from molecular dynamics (MD) simulation.

## visco.py

Calculates the viscosity from the pressure fluctuations data from an NVT MD simulation using the Einstein and Green-Kubo expressions.
The viscosity is computed from the integral of the elements of the pressure tensor or their autocorrelation function.


Usage: `python visco.py -h`

An example data file 'press.data' is available in example directory. The required values to run the example can be found in 'md.param' file.

## vacf.py 

Calculates self-diffusion coefficients from particle velocietis from molecular dynamics (MD) simulations. 
The self-diffusion coefficients are computed from velocity auto-correlation functions (VACF) using the Green-Kubo expression.
