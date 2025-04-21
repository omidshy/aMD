#!/usr/bin/env python

''' ----------------------------------------------------------------------------------------
vacf.py is a code for calculating self-diffusion coefficients from molecular 
dynamics (MD) simulations. The self-diffusion coefficients are computed from 
velocity auto-correlation functions (VACF) using the Green-Kubo expression.

Open-source free software under GNU GPL v3
Copyright (C) 2022-2025 Omid Shayestehpour

Please cite: J. Phys. Chem. B 2022, 126, 18, 3439–3449. (DOI 10.1021/acs.jpcb.1c10671)
---------------------------------------------------------------------------------------- '''

import sys, os
import numpy as np
import pandas as pd
from matplotlib import pyplot
from scipy import integrate
from scipy.optimize import curve_fit
from tqdm import trange

# --------------------------------------------------------
# Define VACF using FFT
def acf(velocities):
    particles = velocities.shape[0]
    steps = velocities.shape[2]
    lag = steps // 3

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
    timestep = timestep * 10**(-12)
    integral = integrate.cumtrapz(y=vacf/3, dx=timestep, initial=0)

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

    # save the running integral of VACF and the fitted curve as CSV files
    df = pd.DataFrame({"time [ps]" : time[:integral.shape[0]]})
    df['self-diffusion coefficient [m^2/s]'] = integral[:]
    df['fit'] = func(time[:integral.shape[0]], *opt)
    df.to_csv("sd.csv", index=False)

    # save fitting results (i.e. self-diffusion coefficients) to a file
    with open('sd.out', 'w') as out:
        out.write('exponential fit: ' + '\n')
        out.write('  D(t) = %e + %e * exp(-t * %e) (units: D = [m^2/s], t = [s])' % tuple(opt) + '\n')
        out.write('  R^2 = %f   (correlation coefficient)' % Rsqrd + '\n\n')
        out.write('Diffusion coefficient = %f pm^2/ps  = %e m^2/s' % (opt[0]*(10**12), opt[0]) + '\n')

    return integral, opt, func

# Save the VACF as a CSV file
def save_vacf(vacf, time):
    df = pd.DataFrame({"time [ps]" : time[:vacf.shape[0]], "VACF" : vacf/vacf[0]})
    df.to_csv("vacf.csv", index=False)

# Plot the VACF
def plot_vacf(vacf, time):
    pyplot.figure(figsize=(10,5))
    pyplot.plot(time[:vacf.shape[0]], vacf/vacf[0], label='vacf')
    pyplot.xlabel('time [ps]')
    pyplot.ylabel('⟨v(0).v(t)⟩')
    pyplot.legend()
    pyplot.show()

# Plot the running integral of VACF and fitted curve
def plot_diffusion(integral, opt, func, time):
    time = time[:integral.shape[0]]
    pyplot.figure(figsize=(10,5))
    pyplot.plot(time, integral, label='self-diffusion')
    pyplot.plot(time, func(time, *opt), label='fit', linestyle='dashed')
    pyplot.xlabel('time [ps]')
    pyplot.ylabel('D [m^2/s]')
    pyplot.legend()
    pyplot.show()

# -----------------------------------------------------
if __name__ == "__main__":
    '''
    Provide the particle velocity vectors as a NumPy array of shape (number of particles, 3, number of steps).
    Note: shape of the axis=1 is (3) for 3 components of the velocity vector [vx vy vz].
    Note: velocities should be in [m/s].
    Replace the example data bellow with your own data.
    '''
    velocities = np.load('example/velocities')

    # The physical timestep between successive steps in your velocities array in [ps]
    timestep = 0.01

    # Create a time array from timestep and number of steps
    num_steps = velocities.shape[2]
    time = np.linspace(0, num_steps*timestep, num=num_steps, dtype=np.float32, endpoint=False)

    # Compute VACF
    myvacf = acf(velocities)

    # Compute self-diffusion coefficient
    integral, opt, func = diffusion(myvacf, time, timestep)
    print(f'diffusion coefficient = {opt[0]*(10**12):.6f} pm^2/ps  = {opt[0]:.4e} m^2/s')

    # Plot the VACF
    plot_vacf(myvacf, time)

    # Plot the running integral of VACF and fitted curve
    plot_diffusion(integral, opt, func, time)

    # Save the VACF
    save_vacf(myvacf, time)
