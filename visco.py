#!/usr/bin/env python

''' ----------------------------------------------------------------------------------------
visco.py is a code for calculating viscosity from molecular dynamics (MD) simulations.

Open-source free software under GNU GPL v3
Copyright (C) 2022-2025 Omid Shayestehpour

Calculation of viscosity using the Einstein or Green-Kubo expressions.  
Viscosity is determined from the integral of the pressure tensor elements  
or their autocorrelation function, obtained from NVT MD simulations.

Notice: the pressure tensor file should have space-separated columns 
of the following order and units of [atm/bar/Pa]:
Pxx, Pyy, Pzz, Pxy, Pxz, Pyz
---------------------------------------------------------------------------------------- '''

import sys, argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot
from scipy import integrate
from scipy.constants import Boltzmann
from scipy.optimize import curve_fit
from tqdm import trange

# --------------------------------------------------------
# Define conversion ratios from various pressure units to Pascals (Pa)
unit_conversion_ratios = {
    'Pa': 1,
    'atm': 101325,
    'bar': 100000,
}

# Parses command-line arguments
def parse_arguments():

    parser = argparse.ArgumentParser(
        prog="visco.py",
        description='Calculation of viscosity from (NVT) molecular dynamics simulations.'
    )

    # --- Input File Arguments ---
    parser.add_argument(
        'datafile',
        # The 'datafile' attribute in the returned 'args' will be an open file object
        type=argparse.FileType('r'),
        help='Path/name of the pressure tensor data file. '
             'Note: the pressure tensor file should have space-separated columns '
             '(Pxx, Pyy, Pzz, Pxy, Pxz, Pyz) and units of (atm, bar or Pa).'
    )
    parser.add_argument(
        '-u', '--unit',
        choices=unit_conversion_ratios.keys(), # Use keys from the dictionary
        default='atm',
        help='Unit of the provided pressure data: Pa, atm or bar. Default: %(default)s'
    )

    # --- Simulation Parameter Arguments ---
    parser.add_argument(
        '-s', '--steps',
        type=int,
        required=True,
        help='Number of steps to read from the pressure tensor file.'
    )
    parser.add_argument(
        '-t', '--timestep',
        type=float,
        required=True,
        help='Physical timestep between successive pressure data points in [ps].'
    )
    parser.add_argument(
        '-T', '--temperature',
        type=float,
        required=True,
        help='Temperature of the MD simulation in [K].'
    )
    parser.add_argument(
        '-v', '--volume',
        type=float,
        required=True,
        help='Volume of the simulation box in [A^3].'
    )

    # --- Calculation/Output Arguments ---
    parser.add_argument(
        '-d', '--no-diag',
        dest='use_diag',
        action='store_false',
        help='Do *not* include diagonal elements of the pressure tensor for Green-Kubo. '
             'By default, diagonal elements are included.'
    )
    parser.add_argument(
        '-p', '--plot',
        action='store_true',
        help='Show plots of auto-correlation functions and running integral. Default: False.'
    )
    parser.add_argument(
        '-e', '--each',
        type=int,
        default=100,
        metavar='INTERVAL', # Adds placeholder text in help message
        help='Steps interval to save the time evolution of the viscosity. Default: %(default)s steps.'
    )

    args = parser.parse_args()

    return args

# Define autocorrelation
def acf_(data):
    steps = data.shape[0]
    size = steps // 2

    autocorrelation = np.zeros(size, dtype=float)
    for shift in trange(size, ncols=100, desc='Progress'):
            autocorrelation[shift] = np.mean( (data[:steps-shift]) * (data[shift:]) )

    return autocorrelation

# Define autocorrelation using FFT
def acf(data):
    steps = data.shape[0]
    lag = steps // 2

    # Nearest size with power of 2 (for efficiency) to zero-pad the input data
    size = 2 ** np.ceil(np.log2(2 * steps - 1)).astype('int')

    # Compute the FFT
    FFT = np.fft.fft(data, size)

    # Get the power spectrum
    PWR = FFT.conjugate() * FFT

    # Calculate the auto-correlation from inverse FFT of the power spectrum
    COR = np.fft.ifft(PWR)[:steps].real

    autocorrelation = COR / np.arange(steps, 0, -1)

    return autocorrelation[:lag]

# Viscosity from Einstein relation
def einstein(Pxx, Pyy, Pzz, Pxy, Pxz, Pyz, time_array, timestep, volume, temperature):
    '''
    Calculate the viscosity from the Einstein relation
    by integrating the components of the pressure tensor
    '''
    Pxxyy = (Pxx - Pyy) / 2
    Pyyzz = (Pyy - Pzz) / 2

    Pxy_int = integrate.cumulative_trapezoid(y=Pxy, dx=timestep, initial=0)
    Pxz_int = integrate.cumulative_trapezoid(y=Pxz, dx=timestep, initial=0)
    Pyz_int = integrate.cumulative_trapezoid(y=Pyz, dx=timestep, initial=0)

    Pxxyy_int = integrate.cumulative_trapezoid(y=Pxxyy, dx=timestep, initial=0)
    Pyyzz_int = integrate.cumulative_trapezoid(y=Pyyzz, dx=timestep, initial=0)

    integral = (Pxy_int**2 + Pxz_int**2 + Pyz_int**2 + Pxxyy_int**2 + Pyyzz_int**2) / 5

    viscosity = integral[1:] * (volume * 10**(-30) / (2 * Boltzmann * temperature * time_array[1:] * 10**(-12)))

    return viscosity

# Viscosity from Green-Kubo relation
def green_kubo(Pxx, Pyy, Pzz, Pxy, Pxz, Pyz, timestep, volume, temperature, use_diag):
    # Calculate the ACFs
    Pxy_acf = acf(Pxy)
    Pxz_acf = acf(Pxz)
    Pyz_acf = acf(Pyz)

    # Calculate the shear components of the pressure tensor and their ACF
    if use_diag:
        Pxxyy = (Pxx - Pyy) / 2
        Pyyzz = (Pyy - Pzz) / 2
        Pxxzz = (Pxx - Pzz) / 2

        Pxxyy_acf = acf(Pxxyy)
        Pyyzz_acf = acf(Pyyzz)
        Pxxzz_acf = acf(Pxxzz)

    if use_diag:
        avg_acf = (Pxy_acf + Pxz_acf + Pyz_acf + Pxxyy_acf + Pyyzz_acf + Pxxzz_acf) / 6
    else:
        avg_acf = (Pxy_acf + Pxz_acf + Pyz_acf) / 3

    # Integrate the average ACF to get the viscosity
    integral = integrate.cumulative_trapezoid(y=avg_acf, dx=timestep, initial=0)
    viscosity = integral * (volume * 10**(-30) / (Boltzmann * temperature))

    return avg_acf, viscosity

# --------------------------------------------------------
if __name__ == "__main__":
    # Parse the command-line arguments
    args = parse_arguments()

    # Print the parsed command-line arguments
    print(f"Using pressure data file: {args.datafile.name}")
    print(f"Pressure unit: {args.unit}")
    print(f"Number of steps: {args.steps}")
    print(f"Timestep: {args.timestep} ps")
    print(f"Temperature: {args.temperature} K")
    print(f"Volume: {args.volume} A^3")
    print(f"Include diagonal pressure tensor elements: {args.use_diag}")
    print(f"Generate plots: {args.plot}")
    print(f"Save interval: {args.each} steps")

    # Get the conversion ratio using the dictionary lookup
    conv_ratio = unit_conversion_ratios[args.unit]

    # Generate a time array
    end_step = args.steps * args.timestep
    time_array = np.linspace(0, end_step, num=args.steps, endpoint=False)

    # Convert the timestep to [s]
    timestep = args.timestep * 10**(-12)

    # Initiate the pressure tensor component lists
    Pxx, Pyy, Pzz, Pxy, Pxz, Pyz = [], [], [], [], [], []

    # Read the pressure tensor elements from data file
    print('\nReading the pressure tensor data file')
    with args.datafile as file:
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

    # ------------------------------------
    viscosity = einstein(
        Pxx,
        Pyy,
        Pzz,
        Pxy,
        Pxz,
        Pyz,
        time_array,
        timestep,
        args.volume,
        args.temperature
        )

    # Plot the running integral of viscosity
    if args.plot:
        pyplot.figure()
        pyplot.plot(
            time_array[:viscosity.shape[0]],
            viscosity[:]*1000, label='Viscosity'
            )
        pyplot.xlabel('Time (ps)')
        pyplot.ylabel('Viscosity (mPa.s)')
        pyplot.legend()
        pyplot.show()

    # Save the running integral of viscosity as a csv file
    df = pd.DataFrame({
        "time(ps)" : time_array[:viscosity.shape[0]:args.each],
        "viscosity(Pa.s)" : viscosity[::args.each]
        })
    df.to_csv("viscosity_Einstein.csv", index=False)

    print(f"\nViscosity (Einstein): {round((viscosity[-1] * 1000), 2)} [mPa.s]")

    # ------------------------------------
    avg_acf, viscosity = green_kubo(
        Pxx,
        Pyy,
        Pzz,
        Pxy,
        Pxz,
        Pyz,
        timestep,
        args.volume,
        args.temperature,
        args.use_diag
        )

    # Plot the normalized average ACF
    if args.plot:
        norm_avg_acf = avg_acf / avg_acf[0]
        pyplot.figure()
        pyplot.plot(
            time_array[:avg_acf.shape[0]],
            norm_avg_acf[:], label='Average'
            )
        pyplot.xlabel('Time (ps)')
        pyplot.ylabel('ACF')
        pyplot.legend()
        pyplot.show()

    # Save the normalized average ACF as a csv file
    norm_avg_acf = avg_acf / avg_acf[0]
    df = pd.DataFrame({
        "time (ps)" : time_array[:avg_acf.shape[0]],
        "ACF" : norm_avg_acf[:]
        })
    df.to_csv("avg_acf.csv", index=False)

    # Plot the time evolution of the viscosity estimate
    if args.plot:
        pyplot.figure()
        pyplot.plot(
            time_array[:viscosity.shape[0]],
            viscosity[:]*1000, label='Viscosity'
            )
        pyplot.xlabel('Time (ps)')
        pyplot.ylabel('Viscosity (mPa.s)')
        pyplot.legend()
        pyplot.show()

    # Save running integral of the viscosity as a csv file
    df = pd.DataFrame({
        "time(ps)" : time_array[:viscosity.shape[0]:args.each],
        "viscosity(Pa.s)" : viscosity[::args.each]
        })
    df.to_csv("viscosity_GK.csv", index=False)

    print(f"Viscosity (Green-Kubo): {round((viscosity[-1] * 1000), 2)} [mPa.s]")
    print("Note: Do not trust these values!")
    print("You should fit an exponential function to the running integral and take its limit.")
