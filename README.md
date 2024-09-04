
# Python codes for MD simulation analysis

A set of programs to compute physical properties from molecular dynamics (MD) simulations.

## visco.py

Calculate the viscosity using components of the pressure tensor from a canonical ensemble (NVT) MD simulation using the Einstein/Green-Kubo relation, 
where the viscosity is computed from the integral of the pressure tensor elements or their autocorrelation function.


Usage: `python visco.py -h`

An example data-file 'press.data' is available in the example directory. The required values to run the example can be found in 'md.param' file.

## vacf.py 

Calculate self-diffusion coefficients from particle velocities. 
The self-diffusion coefficients are computed from velocity auto-correlation functions (VACF) using the Green-Kubo relation.
