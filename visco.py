# --------------------------------------------------------------------------------------------
#
# visco.py is a code for calculating viscosity from molecular dynamics (MD) simulations.
# Copyright (C) 2021 Omid Shayestehpour
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
# 
# Calculation of viscosity using the Einstein or Green-Kubo expressions.
# Viscosity is computed from the integral of the elements of the pressure tensor
# (or their auto-correlation function) collected from NVT MD simulations.
#
# Notice: the pressure tensor file should have space-separated columns 
# of the following order and units of [atm/bar/Pa]:
# Pxx, Pyy, Pzz, Pxy, Pxz, Pyz
#
# --------------------------------------------------------------------------------------------

import sys, argparse
import pylab
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.constants import Boltzmann
from scipy.optimize import curve_fit
from tqdm import trange

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define ACF
def acf(data):
    N = data.shape[0]
    size = N//2

    autocorrelation = np.zeros(size, dtype=float)
    for shift in trange(size, ncols=100, desc='Progress'):
            autocorrelation[shift] = np.mean( (data[:N-shift]) * (data[shift:]) )

    return autocorrelation

# Define ACF using FFT
def acf_fft(data):
    N = data.shape[0]
    lag = N//2

    # Nearest size with power of 2 (for efficiency) to zero-pad the input data
    size = 2 ** np.ceil(np.log2(2*N - 1)).astype('int')

    # Compute the FFT
    FFT = np.fft.fft(data, size)

    # Get the power spectrum
    PWR = FFT.conjugate() * FFT

    # Calculate the auto-correlation from inverse FFT of the power spectrum
    CORR = np.fft.ifft(PWR)[:N].real

    autocorrelation = CORR / np.arange(N, 0, -1)

    return autocorrelation[:lag]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def parser():
    parser = argparse.ArgumentParser(prog="visco.py", description='Calculation of viscosity from (NVT) molecular dynamics simulations.')

    parser.add_argument('datafile', 
                        help='the path/name of the pressure tensor data file. \
                        Note: the pressure tensor file should have space-separated columns \
                        (of the order Pxx, Pyy, Pzz, Pxy, Pxz, Pyz) and units of (atm, bar or Pa)')

    parser.add_argument('-u', '--unit', choices=['Pa', 'atm', 'bar'], default='atm', 
                        help='unit of the provided pressure data: Pa, atm or bar. default = atm')

    parser.add_argument('-s', '--steps', type=int, required=True, 
                        help='number of steps to read from the pressure tensor file')

    parser.add_argument('-t', '--timestep', type=float, required=True, 
                        help='the physical timestep between two successive pressure data in your file in [ps]')

    parser.add_argument('-T', '--temperature', type=float, required=True, 
                        help='the temperature of your MD simulation in [K]')

    parser.add_argument('-v', '--volume', type=float, required=True, 
                        help='the volume of your simulation box in [A^3]')

    parser.add_argument('-d', '--diag', action='store_false',  
                        help='also include the diagonal elements of the pressure tensor for viscosity \
                        calculation using Green-Kubo approach. default = True')

    parser.add_argument('-p', '--plot', action='store_true',  
                        help='show the plots of auto-correlation functions and \
                        running integral of the viscosity. default = False')

    parser.add_argument('-e', '--each', type=int, default=100, 
                        help='the steps interval to save the time evolution of the viscosity. default = 100')

    args = parser.parse_args()

    try:
        with open(args.datafile, "r") as file:
            pass
    except IOError as error:
        print(f'Error: {error}')
        sys.exit(1)

    return args

args = parser()

# Conversion ratio from atm/bar to Pa
if args.unit == 'Pa':
    conv_ratio = 1
elif args.unit == 'atm':
    conv_ratio = 101325
elif args.unit == 'bar':
    conv_ratio = 100000

# Calculate the kBT value
kBT = Boltzmann * args.temperature

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initiate the pressure tensor component lists
Pxx, Pyy, Pzz, Pxy, Pxz, Pyz = [], [], [], [], [], []

# Read the pressure tensor elements from data file
print('\nReading the pressure tensor data file')
with open(args.datafile, "r") as file:
    for _ in trange(args.steps, ncols=100, desc='Progress'):
        line = file.readline()
        step = list(map(float, line.split()))
        Pxx.append(step[0]*conv_ratio)
        Pyy.append(step[1]*conv_ratio)
        Pzz.append(step[2]*conv_ratio)
        Pxy.append(step[3]*conv_ratio)
        Pxz.append(step[4]*conv_ratio)
        Pyz.append(step[5]*conv_ratio)

# Convert lists to numpy arrays
Pxx = np.array(Pxx)
Pyy = np.array(Pyy)
Pzz = np.array(Pzz)
Pxy = np.array(Pxy)
Pxz = np.array(Pxz)
Pyz = np.array(Pyz)

# Generate the time array
end_step = args.steps * args.timestep
Time = np.linspace(0, end_step, num=args.steps, endpoint=False)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Viscosity from Einstein relation
def einstein():

    Pxxyy = (Pxx - Pyy) / 2
    Pyyzz = (Pyy - Pzz) / 2

    '''
    Calculate the viscosity from the Einstein relation 
    by integrating the components of the pressure tensor
    '''
    timestep = args.timestep * 10**(-12)

    Pxy_int = integrate.cumtrapz(y=Pxy, dx=timestep, initial=0)
    Pxz_int = integrate.cumtrapz(y=Pxz, dx=timestep, initial=0)
    Pyz_int = integrate.cumtrapz(y=Pyz, dx=timestep, initial=0)

    Pxxyy_int = integrate.cumtrapz(y=Pxxyy, dx=timestep, initial=0)
    Pyyzz_int = integrate.cumtrapz(y=Pyyzz, dx=timestep, initial=0)

    integral = (Pxy_int**2 + Pxz_int**2 + Pyz_int**2 + Pxxyy_int**2 + Pyyzz_int**2) / 5

    viscosity = integral[1:] * ( args.volume * 10**(-30) / (2 * kBT * Time[1:] * 10**(-12)) )

    return viscosity

viscosity = einstein()

print(f"\nViscosity (Einstein): {round((viscosity[-1] * 1000), 2)} [mPa.s]")

# Plot the running integral of viscosity
if args.plot:
    pylab.figure()
    pylab.plot(Time[:viscosity.shape[0]], viscosity[:]*1000, label='Viscosity')
    pylab.xlabel('Time (ps)')
    pylab.ylabel('Viscosity (mPa.s)')
    pylab.legend()
    pylab.show()

# Save the running integral of viscosity as a csv file
df = pd.DataFrame({"time(ps)" : Time[:viscosity.shape[0]:args.each], "viscosity(Pa.s)" : viscosity[::args.each]})
df.to_csv("viscosity_Einstein.csv", index=False)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Viscosity from Green-Kubo relation
def green_kubo():
    # Calculate the ACFs
    Pxy_acf = acf_fft(Pxy)
    Pxz_acf = acf_fft(Pxz)
    Pyz_acf = acf_fft(Pyz)

    # Calculate the shear components of the pressure tensor and their ACF
    if args.diag:
        Pxxyy = (Pxx - Pyy) / 2
        Pyyzz = (Pyy - Pzz) / 2
        Pxxzz = (Pxx - Pzz) / 2

        Pxxyy_acf = acf_fft(Pxxyy)
        Pyyzz_acf = acf_fft(Pyyzz)
        Pxxzz_acf = acf_fft(Pxxzz)

    if args.diag:
        avg_acf = (Pxy_acf + Pxz_acf + Pyz_acf + Pxxyy_acf + Pyyzz_acf + Pxxzz_acf) / 6
    else:
        avg_acf = (Pxy_acf + Pxz_acf + Pyz_acf) / 3

    # Integrate the average ACF to get the viscosity
    timestep = args.timestep * 10**(-12)
    integral = integrate.cumtrapz(y=avg_acf, dx=timestep, initial=0)
    viscosity = integral * (args.volume * 10**(-30) / kBT)

    return avg_acf, viscosity

avg_acf, viscosity = green_kubo()

# Plot the normalized average ACF
if args.plot:
    norm_avg_acf = avg_acf / avg_acf[0]
    pylab.figure()
    pylab.plot(Time[:avg_acf.shape[0]], norm_avg_acf[:], label='Average')
    pylab.xlabel('Time (ps)')
    pylab.ylabel('ACF')
    pylab.legend()
    pylab.show()

# Save the normalized average ACF as a csv file
norm_avg_acf = avg_acf / avg_acf[0]
df = pd.DataFrame({"time (ps)" : Time[:avg_acf.shape[0]], "ACF" : norm_avg_acf[:]})
df.to_csv("avg_acf.csv", index=False)

print(f"Viscosity (Green-Kubo): {round((viscosity[-1] * 1000), 2)} [mPa.s]")
print("Note: do not trust these values! You should fit an exponential function to the running integral and take its limit.")

# Plot the time evolution of the viscosity estimate
if args.plot:
    pylab.figure()
    pylab.plot(Time[:viscosity.shape[0]], viscosity[:]*1000, label='Viscosity')
    pylab.xlabel('Time (ps)')
    pylab.ylabel('Viscosity (mPa.s)')
    pylab.legend()
    pylab.show()

# Save running integral of the viscosity as a csv file
df = pd.DataFrame({"time(ps)" : Time[:viscosity.shape[0]:args.each], "viscosity(Pa.s)" : viscosity[::args.each]})
df.to_csv("viscosity_GK.csv", index=False)
