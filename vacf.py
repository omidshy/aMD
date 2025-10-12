"""
----------------------------------------------------------------------------------------
vacf.py is a code for calculating self-diffusion coefficients from molecular 
dynamics (MD) simulations. The self-diffusion coefficients are computed from 
velocity auto-correlation functions (VACF) using the Green-Kubo expression.

Open-source free software under GNU GPL v3
Copyright (C) 2022-2025 Omid Shayestehpour

Please cite: J. Phys. Chem. B 2022, 126, 18, 3439–3449. (DOI 10.1021/acs.jpcb.1c10671)
----------------------------------------------------------------------------------------
"""

import os
import argparse
import numpy as np
import pandas as pd
from itertools import islice
from scipy import integrate
from scipy.optimize import curve_fit
from tqdm import trange
from utils import plot_results

# --------------------------------------------------------
# Define conversion ratios from various velocity units to [m/s]
unit_conversion_ratios = {
    'm/s'   : 1,
    'cm/s'  : 1e-2,
    'A/fs'  : 1e+5,
    'A/ps'  : 1e+2,
}

# Parse command-line arguments
def parse_arguments():

    parser = argparse.ArgumentParser(
        prog="vacf.py",
        description='Calculates self-diffusion coefficients from particle velocities.'
    )

    parser.add_argument(
        'datafile',
        # The 'datafile' attribute in the returned 'args' will be an open file object
        type=argparse.FileType('r'),
        help='Name of the particle velocity data file. '
             'Note: the velocity data file should have space-separated columns of '
             'velocity vector components (vx, vy, vz) and unit of (m/s, cm/s, Å/fs or Å/ps). '
             'The total number of lines should be equal to the (number of steps * number of particles) '
             'and the order of particles in each frame (i.e. step) should be identical.'
    )
    parser.add_argument(
        '-u', '--unit',
        choices=unit_conversion_ratios.keys(), # Use keys from the dictionary
        default='m/s',
        help='Unit of the provided velocity data: m/s, cm/s, A/fs or A/ps. Default: %(default)s'
    )
    parser.add_argument(
        '-s', '--steps',
        type=int,
        required=True,
        help='Number of steps to read from the velocity data file.'
    )
    parser.add_argument(
        '-n', '--particles',
        type=int,
        required=True,
        help='Number of particles in each frame of the velocity data file.'
    )
    parser.add_argument(
        '-t', '--timestep',
        type=float,
        required=True,
        help='Physical timestep between successive velocity data points in [ps].'
    )

    args = parser.parse_args()

    return args

# Define VACF using FFT
def acf(velocities, max_lag):
    """
    Computes the velocity autocorrelation function (VACF) for a set of particles,
    which is a measure of how the velocity of a particle at a given time is
    correlated with its velocity at a later time.

    Parameters
    ----------
    velocities : np.ndarray
        A 3D NumPy array of shape (N, 3, M) where N is the number of particles,
        3 represents the x, y, z dimensions, and M is the number of time steps.
    max_lag : float
        Portion of the total simulation length to be used as maximum lag time.

    Returns
    -------
    vacf : np.ndarray
        A 1D NumPy array representing the average VACF over all particles.

    Notes
    -----
    This implementation leverages the Wiener-Khinchin theorem, which states that
    the autocorrelation of a signal is the inverse Fourier transform of its power
    spectral density. This allows for a fast calculation of the autocorrelation.
    The maximum lag time is set to 30% of the total number of steps, for better
    statistics, as correlations often decay to zero long before this point.
    """
    particles = velocities.shape[0]
    steps = velocities.shape[2]
    lag = int(steps * max_lag)

    # nearest size with power of 2 (for efficiency) to zero-pad the input data
    size = 2 ** np.ceil(np.log2(2*steps - 1)).astype('int')

    vacf = np.zeros((particles, lag), dtype=np.float32)
    for i in trange(particles, ncols=100, desc='Progress'):

        # compute the FFT
        Xfft = np.fft.fft(velocities[i, 0], size)
        Yfft = np.fft.fft(velocities[i, 1], size)
        Zfft = np.fft.fft(velocities[i, 2], size)

        # get the power spectrum
        Xpwr = Xfft.conjugate() * Xfft
        Ypwr = Yfft.conjugate() * Yfft
        Zpwr = Zfft.conjugate() * Zfft

        # calculate the auto-correlation from inverse FFT of the power spectrum
        Xcorr = np.fft.ifft(Xpwr)[:steps].real
        Ycorr = np.fft.ifft(Ypwr)[:steps].real
        Zcorr = np.fft.ifft(Zpwr)[:steps].real

        autocorrelation = (Xcorr + Ycorr + Zcorr) / np.arange(steps, 0, -1)

        vacf[i] = autocorrelation[:lag]

    return np.mean(vacf, axis=0)

# Integrate the VACF and calculate the self-diffusion coefficient using the Green-Kubo relation
def diffusion(vacf, time, timestep):
    """
    Calculates the diffusion coefficient from the velocity autocorrelation function (VACF).

    Parameters
    ----------
    vacf : np.ndarray
        The velocity autocorrelation function. This is a 2D NumPy array
        representing the correlation over time.
    time : np.ndarray
        The time array corresponding to the VACF. This should be a 1D NumPy array
        of the same length as ``vacf``.
    timestep : float
        The time step between consecutive points in the `time` array.

    Returns
    -------
    integral : np.ndarray
        The running integral of the VACF, divided by 3, representing a quantity
        proportional to the mean squared displacement over time.
    func : callable
        The exponential function used for fitting. It has the form :math:`a + b e^{-c x}`.
    opt : np.ndarray
        The optimal parameters (:math:`a`, :math:`b`, :math:`c`) found by the curve fitting.
    Rsqrd : float
        The coefficient of determination (:math:`R^2`) for the exponential fit, indicating
        how well the model fits the data.

    Notes
    -----
    The diffusion coefficient (:math:`D`) is given by the long-time limit of the integral, which
    corresponds to the parameter :math:`a` in the fitted function. Specifically, :math:`D = a`.
    The VACF is divided by 3, assuming a 3-dimensional system, to relate it to the
    mean squared displacement in one dimension.
    """
    integral = integrate.cumulative_trapezoid(y=vacf/3, dx=timestep, initial=0)

    # fitting an exponential function to the running integral
    def func(x, a, b, c):
        return a + b * np.exp(c*(-x))

    # initial guess of the fitting parameters
    initialGuess = [1.0, 1.0, 1.0]  

    # perform curve fitting
    opt, cov = curve_fit(func, time[:integral.shape[0]], integral[:], initialGuess)
    residuals = integral[:] - func(time[:integral.shape[0]], *opt)
    ssRes = np.sum(residuals**2)
    ssTot = np.sum((integral[:] - np.mean(integral[:]))**2)
    Rsqrd = 1 - (ssRes / ssTot)

    return integral, func, opt, Rsqrd

# -----------------------------------------------------
def main():
    # Parse the command-line arguments
    args = parse_arguments()

    # Print the parsed command-line arguments
    print(f"Using velocity data file: {args.datafile.name}")
    print(f"Velocity unit: {args.unit}")
    print(f"Number of steps to read: {args.steps}")
    print(f"Number of particles: {args.particles}")
    print(f"Timestep: {args.timestep} ps")

    # Get the conversion ratio
    conv_ratio = unit_conversion_ratios[args.unit]

    # Generate a time array
    end_step = args.steps * args.timestep
    time_array = np.linspace(0, end_step, num=args.steps, dtype=float, endpoint=False)

    # Convert the timestep to [s]
    timestep = args.timestep * 10**(-12)

    # Initiate the velocity array
    velocities = np.zeros((args.particles, 3, args.steps), dtype=float)

    # Read the particle velocities from data file
    print('\nReading the velocity data file')
    with args.datafile as file:
        for step in trange(args.steps, ncols=100, desc='Progress'):
            frame = [x.strip() for x in islice(file, args.particles)] # Read frame by frame
            for i, line in enumerate(frame):
                velocity = list(map(float, line.split()))
                velocity = [comp * conv_ratio for comp in velocity]
                velocities[i, :, step] = velocity

    # Create a directory to save the results
    if not os.path.exists('results'):
        os.makedirs('results')

    # Compute VACF
    print('\nCalculating velocity auto-correlation function')
    vacf = acf(velocities, 0.3) # using 30% of the total length as max lag time

    # Compute self-diffusion coefficient
    integral, func, opt, Rsqrd = diffusion(vacf, time_array, timestep)

    # Save the VACF as a CSV file
    df = pd.DataFrame({"time [ps]" : time_array[:vacf.shape[0]], "VACF" : vacf/vacf[0]})
    file = os.path.join("results", "vacf.csv")
    df.to_csv(file, index=False)

    # Save the running integral of VACF and the fitted curve as CSV files
    df = pd.DataFrame({"time [ps]" : time_array[:integral.shape[0]]})
    df['self-diffusion coefficient [m^2/s]'] = integral[:]
    df['fit'] = func(time_array[:integral.shape[0]], *opt)
    file = os.path.join("results", "diffusion.csv")
    df.to_csv(file, index=False)

    # Save fitting results (i.e. self-diffusion coefficient) to a file
    file = os.path.join("results", "diffusion.out")
    with open(file, 'w') as out:
        out.write('exponential fit: ' + '\n')
        out.write('  D(t) = %e + %e * exp(-t * %e) (units: D = [m^2/s], t = [s])' % tuple(opt) + '\n')
        out.write('  R^2 = %f   (correlation coefficient)' % Rsqrd + '\n\n')
        out.write('Diffusion coefficient = %f pm^2/ps  = %e m^2/s' % (opt[0]*(10**12), opt[0]) + '\n')

    # Print computed self-diffusion coefficient
    print(f'\ndiffusion coefficient = {opt[0]*(10**12):.6f} pm^2/ps  = {opt[0]:.4e} m^2/s')

    # Plot the VACF
    data = [
        {
        'x': time_array[:vacf.shape[0]],
        'y': vacf/vacf[0],
        'label':'vacf', 'linestyle':'solid', 'color':'#9467bd'
        }
    ]
    labels = {'x': 'Time (ps)','y': '⟨v(0).v(t)⟩'}

    plot_results(data, labels)

    # Plot the running integral of VACF and fitted curve
    time = time_array[:integral.shape[0]]
    data = [
        {
        'x': time,
        'y': integral,
        'label':'self-diffusion', 'linestyle':'solid', 'color':'#9467bd'
        },
        {
        'x': time,
        'y': func(time, *opt),
        'label':'fit', 'linestyle':'dashed', 'color':'red'
        }
    ]
    labels = {'x': 'Time (ps)','y': 'D [m^2/s]'}

    plot_results(data, labels)

if __name__ == "__main__":
    main()