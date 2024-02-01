
# Python codes for MD simulation analysis

This is the repository for a set of programs to calculate physical properties from molecular dynamics (MD) simulation.

## visco.py

Calculates the viscosity from the pressure fluctuations data from an NVT MD simulation using the Einstein and Green-Kubo expressions.
The viscosity is computed from the integral of the elements of the pressure tensor or their autocorrelation function.


Usage: `python visco.py -h`

An example data file 'press.data' is available in example directory. The required values to run the example can be found in 'md.param' file.
