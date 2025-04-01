
# Python codes for MD simulation analysis

<<<<<<< HEAD
A collection of Python scripts for computing physical properties from molecular dynamics (MD) simulations.

## visco.py

Calculate viscosity using components of the pressure tensor obtained from a canonical ensemble (NVT) 
molecular dynamics (MD) simulation. Apply the Einstein or Green-Kubo relation, where viscosity is 
determined from the integral of the pressure tensor elements or their autocorrelation function, respectively.
=======
A set of programs to compute physical properties from molecular dynamics (MD) simulations.

## visco.py

Calculate the viscosity using components of the pressure tensor from a canonical ensemble (NVT) MD simulation using the Einstein/Green-Kubo relation, 
where the viscosity is computed from the integral of the pressure tensor elements or their autocorrelation function.
>>>>>>> 2fb213c0a51f7a4c819acde8ccf36c1ac2cddd19


Usage: `python visco.py -h`

<<<<<<< HEAD
An example data file, press.data, is available in the example directory. The required values to run 
the example can be found in the md.param file.

## vacf.py

Calculate self-diffusion coefficients from particle velocities. The self-diffusion coefficients are 
computed from velocity auto-correlation functions (VACF) using the Green-Kubo relation.
=======
An example data-file 'press.data' is available in the example directory. The required values to run the example can be found in 'md.param' file.

## vacf.py 

Calculate self-diffusion coefficients from particle velocities. 
The self-diffusion coefficients are computed from velocity auto-correlation functions (VACF) using the Green-Kubo relation.
>>>>>>> 2fb213c0a51f7a4c819acde8ccf36c1ac2cddd19
