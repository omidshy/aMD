#!/usr/bin/env python

''' ----------------------------------------------------------------------------------------
vacf.py is a code for calculating self-diffusion coefficients from molecular 
dynamics (MD) simulations. The self-diffusion coefficients are computed from 
velocity auto-correlation functions (VACF) using the Green-Kubo expression.

Open-source free software under GNU GPL v3
Copyright (C) 2022-2024 Omid Shayestehpour

Please cite: J. Phys. Chem. B 2022, 126, 18, 3439–3449. (DOI 10.1021/acs.jpcb.1c10671)
---------------------------------------------------------------------------------------- '''

import sys, os
import numpy as np
import pandas as pd
from matplotlib import pyplot
from scipy import integrate
from scipy.optimize import curve_fit
from tqdm import trange

def vacf(velocities):
    # define VACF using FFT
    
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

def diffusion(vacf, time, timestep):
    # integrate the VACF and calculate the self-diffusion coefficient from Green-Kubo relation

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
    df = pd.DataFrame({"time (ps)" : time[:integral.shape[0]], "self-diffusion coefficient (m^2/s)" : integral[:]})
    df.to_csv("sd.csv", index=False)
    df = pd.DataFrame({"time (ps)" : time[:integral.shape[0]], "self-diffusion coefficient (m^2/s)" : func(time[:integral.shape[0]], *opt)})
    df.to_csv("fit.csv", index=False)

    # save fitting results (i.e. self-diffusion coefficients) to a file
    with open('sd.out', 'w') as out:
        out.write('exponential fit: ' + '\n')
        out.write('  D(t) = %e + %e * exp(-t * %e)   (units: [D] = m^2/s, [t] = s)' % tuple(opt) + '\n')
        out.write('  R^2 = %f   (correlation coefficient)' % Rsqrd + '\n\n')
        out.write('Diffusion coefficient = %f pm^2/ps  = %e m^2/s' % (opt[0]*(10**12), opt[0]) + '\n\n')

    return opt[0]

def save_vacf(vacf, time):
    # save the VACF (and normalized VACF) as a CSV file
    norm_vacf = vacf / vacf[0]
    df = pd.DataFrame({"time (ps)" : time[:vacf.shape[0]], "VACF" : vacf[:], "normalized VACF" : norm_vacf[:]})
    df.to_csv("vacf.csv", index=False)

def plot_vacf(vacf, time):
    # plot the normalized VACF
    norm_vacf = vacf / vacf[0]
    pyplot.figure()
    pyplot.plot(time[:vacf.shape[0]], norm_vacf[:], label='vacf')
    pyplot.xlabel('time (ps)')
    pyplot.ylabel('VACF')
    pyplot.legend()
    pyplot.show()

# -----------------------------------------------------
'''
provide the particle velocity vectors as a NumPy array of shape (number of particles, 3, number of steps).
note: shape of the axis=1 is (3) for 3 components of the velocity vector [vx vy vz].
note: velocities should be in [m/s].
replace the random data bellow with your actual data.
'''
velocities = np.random.normal(0, 0.1, size=(100, 3, 1000))

# the physical timestep between successive steps in your velocities array in [ps]
timestep = 0.01

# we create a time array from timestep and number of steps
num_steps = velocities.shape[2]
time = np.linspace(0, num_steps*timestep, num=num_steps, endpoint=False)

# we compute VACF
myvacf = vacf(velocities)

# we compute self-diffusion coefficient
sd = diffusion(myvacf, time, timestep)
print(f'\ndiffusion coefficient = {sd*(10**12):.6f} pm^2/ps  = {sd:.4e} m^2/s')

# we plot the VACF
plot_vacf(myvacf, time)

# we save the VACF
save_vacf(myvacf, time)
