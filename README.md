
# Python codes for MD simulation analysis

A collection of Python scripts for computing physical properties and analyzing trajectories from
molecular dynamics (MD) simulations.

## visco.py

Calculates viscosity using components of the pressure tensor obtained from a canonical ensemble (NVT)
molecular dynamics (MD) simulation. Employs the Einstein or Green-Kubo relation, where viscosity is
determined from the integral of the pressure tensor elements or their auto-correlation function, respectively.

The viscosity, *η*, is calculated from the integral of the pressure tensor auto-correlation
function over time following the Green--Kubo approach

<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="assets/visco_gk_dark.png"
  />
  <source
    media="(prefers-color-scheme: light)"
    srcset="assets/visco_gk_light.png"
  />
  <img
    alt="viscosity Green--Kubo equation"
    src="assets/visco_gk_light.png"
    height="45"
  />
</picture>

<!--$$
\eta = \frac{V}{k_B T} \int_0^\infty \left\langle P_{\alpha \beta} \left( t \right)
\cdot P_{\alpha \beta} \left( t_0 \right) \right\rangle dt
$$-->

or the Einstein approach

<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="assets/visco_en_dark.png"
  />
  <source
    media="(prefers-color-scheme: light)"
    srcset="assets/visco_en_light.png"
  />
  <img
    alt="viscosity Einstein equation"
    src="assets/visco_en_light.png"
    height="55"
  />
</picture>

<!--$$
\eta = \lim_{t \to \infty} \frac{V}{2 t k_B T}
\left\langle \left( \int_0^\infty P_{\alpha \beta}(t') dt' \right)^2  \right\rangle
$$-->

where *V* is the simulation box volume, *k_B* is the Boltzmann constant, *T* is temperature,
*Pαβ* denotes the off-diagonal element *αβ* of the pressure tensor,
and the brackets indicate that average must be taken over all time origins *t0*.

Usage: `python visco.py -h`

An example data file, press.data, is available in the example directory. The required values to run 
the example can be found in the md.param file.

## vacf.py

Calculates self-diffusion coefficient from particle velocities. The self-diffusion coefficient, *D*, is
computed from an exponential fit to the running integral of velocity auto-correlation function (VACF)
using the following Green-Kubo relation

<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="assets/sdc_gk_dark.png"
  />
  <source
    media="(prefers-color-scheme: light)"
    srcset="assets/sdc_gk_light.png"
  />
  <img
    alt="diffusion Green--Kubo equation"
    src="assets/sdc_gk_light.png"
    height="45"
  />
</picture>

<!--$$
D = \frac{1}{3} \int_0^\infty \left\langle \mathbf{v}_i(t) \cdot \mathbf{v}_i(t_0) \right\rangle dt
$$-->

where ***v**i(t)* denotes the velocity of particle *i* at any specific time *t*.

Usage: `python vacf.py -h`

An example data file, velocity.data, is available in the example directory. The required values to run
the example can be found in the md.param file.